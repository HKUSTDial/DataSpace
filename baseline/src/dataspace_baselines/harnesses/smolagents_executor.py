from __future__ import annotations

from contextlib import closing
import secrets
import time

from smolagents.monitoring import LogLevel
from smolagents.remote_executors import (
    DockerExecutor,
    RemotePythonExecutor,
    _create_kernel_http,
    _websocket_run_code_raise_errors,
)
from websocket import create_connection


class PrebuiltDockerExecutor(DockerExecutor):
    """Use the official executor with packages baked into a pinned image."""

    def install_packages(self, additional_imports: list[str]):
        return list(additional_imports)


class InternalDockerExecutor(PrebuiltDockerExecutor):
    """Connect to a prebuilt executor through an internal Docker bridge.

    Docker does not publish host ports for an ``internal`` network. The host
    can still reach the container's bridge address, so this class changes only
    the official executor's bootstrap connection. Code execution, transport,
    serialization, and cleanup continue to use ``DockerExecutor`` methods.
    """

    def _warm_up_kernel(self) -> None:
        """Wait for a usable multiplexed channel before installing tools.

        A newly created kernel can accept a websocket before its IOPub channel
        is ready. In that race the official receive loop sees the shell reply
        but can wait indefinitely for the matching idle event. ``pass`` is
        safe to retry on a fresh connection until all channels are available.
        """

        last_error: Exception | None = None
        for _ in range(3):
            try:
                with closing(create_connection(self.ws_url, timeout=10)) as ws:
                    _websocket_run_code_raise_errors(
                        "pass", ws, self.logger, self.allow_pickle
                    )
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.25)
        raise RuntimeError("executor kernel did not become ready") from last_error

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        image_name: str = "dataspace-smolagents:1.0",
        container_run_kwargs: dict | None = None,
    ):
        # Call RemotePythonExecutor.__init__ while intentionally bypassing the
        # DockerExecutor bootstrap that requires a published localhost port.
        RemotePythonExecutor.__init__(
            self,
            additional_imports,
            logger,
            allow_pickle,
        )
        try:
            import docker
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Please install the smolagents Docker extra"
            ) from exc

        self.image_name = image_name
        self.client = docker.from_env()
        try:
            self.client.images.get(self.image_name)
            self.logger.log(
                f"Using existing Docker image: {self.image_name}",
                level=LogLevel.INFO,
            )
            container_kwargs = dict(container_run_kwargs or {})
            container_kwargs.pop("ports", None)
            container_kwargs["detach"] = True
            token = secrets.token_urlsafe(16)
            environment = container_kwargs.get("environment") or {}
            if isinstance(environment, list):
                environment = dict(
                    item.split("=", 1) for item in environment if "=" in item
                )
            environment["KG_AUTH_TOKEN"] = token
            container_kwargs["environment"] = environment
            self.container = self.client.containers.run(
                self.image_name,
                **container_kwargs,
            )
            retries = 0
            while self.container.status != "running" and retries < 5:
                time.sleep(1)
                self.container.reload()
                retries += 1
            self.container.reload()
            networks = self.container.attrs["NetworkSettings"]["Networks"]
            addresses = [
                str(network.get("IPAddress", "")).strip()
                for network in networks.values()
            ]
            address = next((value for value in addresses if value), None)
            if address is None:
                raise RuntimeError("executor container has no bridge address")
            self.base_url = f"http://{address}:8888"
            self._wait_for_server(token)
            self.kernel_id = _create_kernel_http(
                f"{self.base_url}/api/kernels?token={token}", self.logger
            )
            self.ws_url = (
                f"ws://{address}:8888/api/kernels/{self.kernel_id}/channels"
                f"?token={token}"
            )
            self._warm_up_kernel()
            self.installed_packages = self.install_packages(additional_imports)
            self.logger.log(
                f"Container {self.container.short_id} is running with "
                f"kernel {self.kernel_id}",
                level=LogLevel.INFO,
            )
        except Exception as exc:
            self.cleanup()
            raise RuntimeError(
                f"Failed to initialize internal Docker executor: {exc}"
            ) from exc
