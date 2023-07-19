import yaml

with open("saint_environment.yml") as file_handle:
    environment_data = yaml.safe_load(file_handle)

with open("requirements_tmp.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        package_name, package_version, source = dependency.split("=")
        if source.startswith("py"):
            file_handle.write("{}=={}\n".format(package_name, package_version))