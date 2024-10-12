import docker

from langchain.tools import tool
from yada.tools import safe_tool, sensitive_tool


@sensitive_tool
@tool
def run_docker_container_image(
    image: str, command: str = None, detach: bool = False
) -> str:
    """
    Run a Docker container image.

    Args:
        image (str): The image to run.
        command (str, optional): The command to run in the container. Defaults to None.
        detach (bool, optional): Whether to run the container in detached mode. Defaults to False.
    """
    client = docker.from_env()
    container_or_logs = client.containers.run(image, command, detach=detach)

    if detach:
        return f"Ran Docker container {container_or_logs.id} from image {image}.\nLOGS\n---\n{container_or_logs.logs()}"
    else:
        return f"Ran Docker container from image {image}.\nLOGS\n---\n{container_or_logs.decode('utf-8')}"


@safe_tool
@tool
def list_all_running_docker_containers() -> str:
    """
    List all running Docker containers.
    """
    client = docker.from_env()
    return "\n".join([str(container) for container in client.containers.list()])


@safe_tool
@tool
def list_all_docker_images() -> str:
    """
    List all Docker images.
    """
    client = docker.from_env()
    return "\n".join([str(image) for image in client.images.list()])


@sensitive_tool
@tool
def build_docker_image_from_dockerfile(directory: str = ".", tag: str = None) -> str:
    """
    Build a Docker image from a Dockerfile.

    Args:
        directory (str, optional): The directory containing the Dockerfile. Defaults to ".".
        tag (str, optional): The tag to assign to the image. Defaults to None.
    """
    client = docker.from_env()
    image, logs = client.images.build(path=directory, tag=tag)
    logz = [log for log in logs]
    return f"""Built Docker image {image.id}.
    Logs: {logz}""".strip()


@safe_tool
@tool
def execute_command_in_docker_container(container_id: str, command: str) -> str:
    """
    Execute a command in a Docker container.

    Args:
        container_id (str): The ID of the Docker container.
        command (str): The command to execute
    """
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
        exit_code, output = container.exec_run(cmd=command)
        return f"Executed command in Docker container {container_id}.\nExit code: {exit_code}\nOutput: {output.decode("utf-8")}"
    except Exception as e:
        return f"An error occurred: {e}"

@safe_tool
@tool
def stop_docker_container(container_id: str) -> str:
    """
    Stop a Docker container.

    Args:
        container_id (str): The ID of the Docker container.
    """
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
        container.stop()
        return f"Stopped Docker container {container_id}."
    except Exception as e:
        return f"An error occurred: {e}"

@sensitive_tool
@tool
def remove_docker_container(container_id: str) -> str:
    """
    Remove a Docker container.

    Args:
        container_id (str): The ID of the Docker container.
    """
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
        container.remove()
        return f"Removed Docker container {container_id}."
    except Exception as e:
        return f"An error occurred: {e}"
    
@sensitive_tool
@tool
def remove_docker_image(image_id: str, force: bool = False) -> str:
    """
    Remove a Docker image.

    Args:
        image_id (str): The ID of the Docker image.
        force (bool, optional): Whether to force removal. Defaults to False.
    """
    try:
        client = docker.from_env()
        client.images.remove(image_id, force=force)
        return f"Removed Docker image {image_id}."
    except Exception as e:
        return f"An error occurred: {e}"
    

@safe_tool
@tool
def docker_logs(container_id: str) -> str:
    """
    Get the logs of a Docker container.

    Args:
        container_id (str): The ID of the Docker container.
    """
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
        return container.logs().decode("utf-8")
    except Exception as e:
        return f"An error occurred: {e}"


@safe_tool
@tool
def docker_compose_up(compose_file: str = "docker-compose.yml") -> str:
    """
    Run `docker-compose up`.

    Args:
        compose_file (str, optional): The Docker Compose file to use. Defaults to "docker-compose.yml".
    """
    try:
        client = docker.from_env()
        client.compose.up(compose_file)
        return "Ran docker-compose up."
    except Exception as e:
        return f"An error occurred: {e}"