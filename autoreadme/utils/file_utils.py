import fnmatch, os
from typing import List

from autoreadme.types import AutodocRepoConfig

def get_environment_variables():
    """Uses .env file to get environment variables for further readme description"""
    if '.env' in os.listdir():
        with open('.env', 'r') as f:
            environments = f.read()
    else:
        environments = ''
    for i in range(1,5):
        environments = environments.replace('\n'*1, '\n')
    environments = '; '.join([env.split('=')[0].strip() for env in environments.split('\n') if len(env)])
    return environments

def should_ignore(file_name: str, ignore: List[str]):
    """Check if the file name matches any of the ignore patterns."""
    return any(fnmatch.fnmatch(file_name, pattern) for pattern in ignore)

def remove_building_files(path: str, config: AutodocRepoConfig) -> None:
    """Remove building json files in the given path."""
    remove_files = ['.'.join(element.split('.')[0:-1])+'.json' for element in os.listdir(path)
                    if (not os.path.isdir(os.path.join(path, element)) and element.split('.')[-1] in ['py', 'yml', 'log', 'txt']
                    and not should_ignore(element, config.ignore))]
    remove_files.append('Dockerfile.json')
    remove_files.append('summary.json')
    remove_files.append('VERSION.json')
    folders = [element for element in os.listdir(path)
               if os.path.isdir(os.path.join(path, element))
               and not should_ignore(element, config.ignore)]
    for file in remove_files:
        try:
            os.remove(os.path.join(path, file))
        except Exception:
            pass
    for folder in folders:
        remove_building_files(os.path.join(path, folder), config)

def get_file_name(input_str, delimiter=".", extension=".md"):
    """Get File Name"""
    last_delimiter_index = input_str.rfind(delimiter)
    if last_delimiter_index == -1:
        # delimiter not found in string
        return input_str + extension
    else:
        return input_str[:last_delimiter_index] + extension


def github_file_url(github_root, input_root, file_path, link_hosted):
    """Get GitHub File URL"""
    root = file_path.replace('\\', '/').replace(input_root, '')
    if link_hosted:
        return f"{github_root}{root}"
    else:
        return f"{github_root}/blob/master/{root}"


def github_folder_url(github_root, input_root, folder_path, link_hosted):
    """Get GitHub Folder URL"""
    root = folder_path.replace('\\', '/').replace(input_root, '')
    if link_hosted:
        return f"{github_root}{root}"
    else:
        return f"{github_root}/tree/master/{root}"
