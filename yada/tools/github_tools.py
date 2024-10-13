from langchain.tools import tool

from yada.tools import sensitive_tool

from git import Repo, GitCommandError


@sensitive_tool
@tool
def clone_github_repository_by_git_url(
    git_url: str, to_path: str = ".", branch: str = "main"
) -> str:
    """
    Clone a GitHub repository by its git URL.

    Args:
        git_url (str): The git URL of the repository.
        to_path (str): The path to clone the repository to.
        branch (str): Optional, The branch to clone, default "main".
    """
    Repo.clone_from(git_url, to_path, branch=branch)
    return f"Cloned the {branch} branch of the repository to {to_path}."


@sensitive_tool
@tool
def checkout_github_repository_branch(branch: str, repository_path: str = ".") -> str:
    """
    Checkout a branch in a GitHub repository. This will create a new branch if it doesn't exist.

    Args:
        branch (str): The branch to checkout.
        repository_path (str): The path to the repository, default ".".
    """
    try:
        repo = Repo(repository_path)
        if branch in repo.heads:
            repo.git.checkout(branch)
            return f"Checked out existing {branch} branch."
        else:
            new_branch = repo.create_head(branch)
            new_branch.checkout()
            return f"Created and checked out new {branch} branch."
    except GitCommandError as e:
        return f"An error occurred: {e}"


@sensitive_tool
@tool
def delete_local_github_repository_branch(
    branch: str, repository_path: str = "."
) -> str:
    """
    Delete a branch in a local GitHub repository.

    Args:
        branch (str): The branch to delete.
        repository_path (str): The path to the repository, default ".".
    """
    try:
        repo = Repo(repository_path)
        repo.git.branch("-D", branch)
        return f"Deleted local branch {branch}."
    except GitCommandError as e:
        return f"An error occurred: {e}"
