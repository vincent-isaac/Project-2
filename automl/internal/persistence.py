import gc
import os
from typing import Dict, Optional

import joblib
from sklearn.pipeline import Pipeline

from automl.utils._dependencies import _check_soft_dependencies
from automl.utils.generic import MLUsecase, get_logger


def deploy_model(
    model, model_name: str, authentication: dict, platform: str = "aws", prep_pipe_=None
):
     

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing deploy_model()")
    logger.info(f"deploy_model({function_params_str})")

    allowed_platforms = ["aws", "gcp", "azure"]

    if platform not in allowed_platforms:
        logger.error(
            f"(Value Error): Platform {platform} is not supported by automl or illegal option"
        )
        raise ValueError(
            f"Platform {platform} is not supported by automl or illegal option"
        )

    if platform:
        if not authentication:
            raise ValueError("Authentication is missing.")

    # general dependencies
    import os

    logger.info("Saving model in active working directory")
    logger.info("SubProcess save_model() called ==================================")
    save_model(model, prep_pipe_=prep_pipe_, model_name=model_name, verbose=False)
    logger.info("SubProcess save_model() end ==================================")

    if platform == "aws":
        logger.info("Platform : AWS S3")

        # checking if boto3 is available
        _check_soft_dependencies("boto3", extra=None, severity="error")
        import boto3

        # initialize s3
        logger.info("Initializing S3 client")
        s3 = boto3.client("s3")
        filename = f"{model_name}.pkl"
        if "path" in authentication:
            key = os.path.join(authentication.get("path"), f"{model_name}.pkl")
        else:
            key = f"{model_name}.pkl"
        bucket_name = authentication.get("bucket")

        if bucket_name is None:
            logger.error(
                "S3 bucket name missing. Provide `bucket` as part of authentication parameter."
            )
            raise ValueError(
                "S3 bucket name missing. Provide `bucket` name as part of authentication parameter."
            )

        import botocore.exceptions

        try:
            s3.upload_file(filename, bucket_name, key)
        except botocore.exceptions.NoCredentialsError:
            logger.error(
                "Boto3 credentials not configured. Refer boto3 documentation "
                "(https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)"
            )
            logger.error("Model deployment to AWS S3 failed.")
            raise ValueError(
                "Boto3 credentials not configured. Refer boto3 documentation "
                "(https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)"
            )
        os.remove(filename)
        print("Model Successfully Deployed on AWS S3")
        logger.info("Model Successfully Deployed on AWS S3")
        logger.info(str(model))

    elif platform == "gcp":
        logger.info("Platform : GCP")

        _check_soft_dependencies(
            "google", extra=None, severity="error", install_name="google-cloud-storage"
        )

        # initialize deployment
        filename = f"{model_name}.pkl"
        key = f"{model_name}.pkl"
        bucket_name = authentication.get("bucket")
        project_name = authentication.get("project")

        if bucket_name is None or project_name is None:
            logger.error(
                "Project and Bucket name missing. Provide `bucket` and `project` as part of "
                "authentication parameter"
            )
            raise ValueError(
                "Project and Bucket name missing. Provide `bucket` and `project` as part of "
                "authentication parameter"
            )

        try:
            _create_bucket_gcp(project_name, bucket_name)
            _upload_blob_gcp(project_name, bucket_name, filename, key)
        except Exception:
            _upload_blob_gcp(project_name, bucket_name, filename, key)
        os.remove(filename)
        print("Model Successfully Deployed on GCP")
        logger.info("Model Successfully Deployed on GCP")
        logger.info(str(model))

    elif platform == "azure":
        logger.info("Platform : Azure Blob Storage")

        _check_soft_dependencies(
            "azure", extra=None, severity="error", install_name="azure-storage-blob"
        )

        # initialize deployment
        filename = f"{model_name}.pkl"
        key = f"{model_name}.pkl"
        container_name = authentication.get("container")

        if container_name is None:
            logger.error(
                "Storage Container name missing. Provide `container` as part of authentication parameter"
            )
            raise ValueError(
                "Storage Container name missing. Provide `container` as part of authentication parameter"
            )

        try:
            _create_container_azure(container_name)
            _upload_blob_azure(container_name, filename, key)
            del container_client
        except Exception:
            _upload_blob_azure(container_name, filename, key)

        os.remove(filename)

        print("Model Successfully Deployed on Azure Storage Blob")
        logger.info("Model Successfully Deployed on Azure Storage Blob")
        logger.info(str(model))

    logger.info(
        "deploy_model() successfully completed......................................"
    )
    gc.collect()


def save_model(
    model,
    model_name: str,
    prep_pipe_=None,
    verbose: bool = True,
    use_case: Optional[MLUsecase] = None,
    **kwargs,
):
  

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing save_model()")
    logger.info(f"save_model({function_params_str})")

    from copy import deepcopy

    logger.info("Adding model into prep_pipe")

    if use_case == MLUsecase.TIME_SERIES:
        from automl.utils.time_series.forecasting.pipeline import (
            _add_model_to_pipeline,
        )

        if prep_pipe_:
            pipeline = deepcopy(prep_pipe_)
            model = _add_model_to_pipeline(pipeline=pipeline, model=model)
        else:
            logger.warning(
                "Only Model saved. Transformations in prep_pipe are ignored."
            )
    else:
        if isinstance(model, Pipeline):
            logger.warning("Only Model saved as it was a pipeline.")
        elif not prep_pipe_:
            logger.warning(
                "Only Model saved. Transformations in prep_pipe are ignored."
            )
        else:
            model_ = deepcopy(prep_pipe_)
            model_.steps.append(("trained_model", model))
            model = model_

    model_name = f"{model_name}.pkl"
    joblib.dump(model, model_name, **kwargs)
    if verbose:
        if prep_pipe_:
            pipe_msg = "Transformation Pipeline and "
        else:
            pipe_msg = ""
        print(f"{pipe_msg}Model Successfully Saved")

    logger.info(f"{model_name} saved in current working directory")
    logger.info(str(model))
    logger.info(
        "save_model() successfully completed......................................"
    )
    gc.collect()
    return model, model_name


def load_model(
    model_name: str,
    platform: Optional[str] = None,
    authentication: Optional[Dict[str, str]] = None,
    verbose: bool = True,
):
     

    function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

    logger = get_logger()

    logger.info("Initializing load_model()")
    logger.info(f"load_model({function_params_str})")

    # exception checking

    if platform:
        if not authentication:
            raise ValueError("Authentication is missing.")

    if not platform:
        model_name = f"{model_name}.pkl"
        model = joblib.load(model_name)
        if verbose:
            print("Transformation Pipeline and Model Successfully Loaded")
        return model

    # cloud providers
    elif platform == "aws":
        import os

        # checking if boto3 is available
        _check_soft_dependencies("boto3", extra=None, severity="error")
        import boto3

        bucketname = authentication.get("bucket")

        if bucketname is None:
            logger.error(
                "S3 bucket name missing. Provide `bucket` as part of authentication parameter"
            )
            raise ValueError(
                "S3 bucket name missing. Provide `bucket` name as part of authentication parameter"
            )

        filename = f"{model_name}.pkl"

        if "path" in authentication:
            key = os.path.join(authentication.get("path"), filename)
        else:
            key = filename

        index = filename.rfind("/")
        s3 = boto3.resource("s3")

        if index == -1:
            s3.Bucket(bucketname).download_file(key, filename)
        else:
            path, key = filename[: index + 1], filename[index + 1 :]
            if not os.path.exists(path):
                os.makedirs(path)
            s3.Bucket(bucketname).download_file(key, filename)

        model = load_model(model_name, verbose=False)

        if verbose:
            print("Transformation Pipeline and Model Successfully Loaded")

        return model

    elif platform == "gcp":
        bucket_name = authentication.get("bucket")
        project_name = authentication.get("project")

        if bucket_name is None or project_name is None:
            logger.error(
                "Project and Bucket name missing. Provide `bucket` and `project` as part of "
                "authentication parameter"
            )
            raise ValueError(
                "Project and Bucket name missing. Provide `bucket` and `project` as part of "
                "authentication parameter"
            )

        filename = f"{model_name}.pkl"

        _download_blob_gcp(project_name, bucket_name, filename, filename)

        model = load_model(model_name, verbose=False)

        if verbose:
            print("Transformation Pipeline and Model Successfully Loaded")
        return model

    elif platform == "azure":
        container_name = authentication.get("container")

        if container_name is None:
            logger.error(
                "Storage Container name missing. Provide `container` as part of authentication parameter"
            )
            raise ValueError(
                "Storage Container name missing. Provide `container` as part of authentication parameter"
            )

        filename = f"{model_name}.pkl"

        _download_blob_azure(container_name, filename, filename)

        model = load_model(model_name, verbose=False)

        if verbose:
            print("Transformation Pipeline and Model Successfully Loaded")
        return model
    else:
        print(f"Platform {platform} is not supported by automl or illegal option")
    gc.collect()


def _create_bucket_gcp(project_name: str, bucket_name: str):
    

    logger = get_logger()

    # bucket_name = "your-new-bucket-name"
    import google.auth.exceptions
    from google.cloud import storage

    try:
        storage_client = storage.Client(project_name)
    except google.auth.exceptions.DefaultCredentialsError:
        logger.error(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )
        raise ValueError(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )

    buckets = storage_client.list_buckets()

    if bucket_name not in buckets:
        bucket = storage_client.create_bucket(bucket_name)
        logger.info("Bucket {} created".format(bucket.name))
    else:
        raise FileExistsError("{} already exists".format(bucket_name))


def _upload_blob_gcp(
    project_name: str,
    bucket_name: str,
    source_file_name: str,
    destination_blob_name: str,
):
    

    logger = get_logger()

    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    import google.auth.exceptions
    from google.cloud import storage

    try:
        storage_client = storage.Client(project_name)
    except google.auth.exceptions.DefaultCredentialsError:
        logger.error(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )
        raise ValueError(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logger.info(
        "File {} uploaded to {}.".format(source_file_name, destination_blob_name)
    )


def _download_blob_gcp(
    project_name: str,
    bucket_name: str,
    source_blob_name: str,
    destination_file_name: str,
):
     

    logger = get_logger()

    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    import google.auth.exceptions
    from google.cloud import storage

    try:
        storage_client = storage.Client(project_name)
    except google.auth.exceptions.DefaultCredentialsError:
        logger.error(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )
        raise ValueError(
            "Environment variable GOOGLE_APPLICATION_CREDENTIALS not set. For more information,"
            " please see https://cloud.google.com/docs/authentication/getting-started"
        )

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if destination_file_name is not None:
        blob.download_to_filename(destination_file_name)

        logger.info(
            "Blob {} downloaded to {}.".format(source_blob_name, destination_file_name)
        )

    return blob


def _create_container_azure(container_name: str):
    

    logger = get_logger()

    # Create the container
    import os

    from azure.storage.blob import BlobServiceClient

    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if connect_str is None:
        logger.error("Environment variable AZURE_STORAGE_CONNECTION_STRING not set")
        raise ValueError("Environment variable AZURE_STORAGE_CONNECTION_STRING not set")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.create_container(container_name)
    return container_client


def _upload_blob_azure(
    container_name: str, source_file_name: str, destination_blob_name: str
):
     

    logger = get_logger()

    from azure.storage.blob import BlobServiceClient

    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if connect_str is None:
        logger.error("Environment variable AZURE_STORAGE_CONNECTION_STRING not set")
        raise ValueError("Environment variable AZURE_STORAGE_CONNECTION_STRING not set")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=destination_blob_name
    )

    # Upload the created file
    with open(source_file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)


def _download_blob_azure(
    container_name: str, source_blob_name: str, destination_file_name: str
):
     

    logger = get_logger()

    import os

    from azure.storage.blob import BlobServiceClient

    connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if connect_str is None:
        logger.error("Environment variable AZURE_STORAGE_CONNECTION_STRING not set")
        raise ValueError("Environment variable AZURE_STORAGE_CONNECTION_STRING not set")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=source_blob_name
    )

    if destination_file_name is not None:
        with open(destination_file_name, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
