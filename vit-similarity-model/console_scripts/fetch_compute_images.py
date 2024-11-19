import time
import requests
from google.cloud import storage
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F
import asyncio
import aiohttp
import sys
import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import torch
import logging

# Set up logging to both the console and a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[
                        logging.FileHandler("/home/ubuntu/google-cloud/fetch_compute_images.log"),
                    ])

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# GLOBAL VARIABLES
client = storage.Client.from_service_account_json(
    # TODO: ADD YOUR PATH TO GOOGLE CLOUD SERVICE ACCOUNT JSON
    r"")
# TODO: ADD YOUR BUCKET NAME
bucket_name = ""
bucket = client.bucket(bucket_name)
processed_image_ids = set()


def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    # TODO: ADD YOUR INSTANCE CONNECTION TO GCLOUD SQL DATABASE
    instance_connection_name = ""
    db_user = ""
    db_pass = "!"
    db_name = ""

    ip_type = IPTypes.PUBLIC

    connector = Connector(ip_type)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool


def check_new_upload(pool) -> list:
    """
    Checks for a new upload in the google cloud storage.
    
    :return: A list of new files.   
    """

    # List all the blobs in the Google Cloud Storage bucket
    blobs = bucket.list_blobs()
    new_files = []

    # Define the SQL query to select image_ids
    select_query = sqlalchemy.text("SELECT image_id FROM inference_results")

    # Connect to the database and execute the query
    with pool.connect() as db_conn:
        result = db_conn.execute(select_query).fetchall()

    # Extract the image_ids from the query result using index 0 (first column)
    image_ids = [row[0] for row in result]

    # Compare blob names with image_ids in the database
    for blob in blobs:
        if blob.name not in image_ids:
            new_files.append(blob.name)
            processed_image_ids.add(blob.name)

    return new_files


async def process_new_files(new_files) -> tuple:
    """
    Async task for processing new files, uploading, inference, and similarity.
    :return: Returns back new files list, and results of inference.
    """

    tasks = []
    for image_id in new_files:
        logging.info(f"[INFO] Processing new image: {image_id}")
        task = asyncio.create_task(send_image_to_inference(image_id))
        tasks.append(task)

    # Wait for all inference tasks to complete
    results = await asyncio.gather(*tasks)

    return new_files, results


def store_inference_result(image_id: str, inference_result: list, pool):
    """
    Stores inference results with SQL query, with unique image_id.
    
    :param image_id: String of unique image id from google cloud storage.
    :param inference_result: List of results from inference of single image.
    """

    # Define the INSERT statement using SQLAlchemy's text()
    insert_stmt = sqlalchemy.text(
        """
        INSERT INTO inference_results (image_id, inference_result)
        VALUES (:image_id, :inference_result)
        """
    )

    # Connect to the database and execute the insert
    with pool.connect() as db_conn:
        db_conn.execute(
            insert_stmt,
            parameters={"image_id": image_id, "inference_result": str(inference_result)}
        )

        # Commit the transaction to save the changes
        db_conn.commit()

    logging.info(f"[INFO] Inference result for image ID {image_id} stored successfully.")


def store_similarity_result(image_id: str, similar_image_id: str, similarity_score: float, pool):
    """
    Updates the similarity result and similar image ID in the database for the given image_id.

    :param image_id: String of unique image ID for which the result will be updated.
    :param similar_image_id: String of the image ID that is most similar.
    :param similarity_score: The float value representing the similarity score.
    """

    # Define the UPDATE statement to add the similarity score and similar image ID
    update_stmt = sqlalchemy.text(
        """
        UPDATE inference_results
        SET similar_image_id = :similar_image_id, initial_similarity = :similarity_score
        WHERE image_id = :image_id
        """
    )

    # Connect to the database and execute the update
    with pool.connect() as db_conn:
        db_conn.execute(
            update_stmt,
            parameters={
                "image_id": image_id,
                "similar_image_id": similar_image_id,
                "similarity_score": similarity_score
            }
        )

        # Commit the transaction to save the changes
        db_conn.commit()

    logging.info(f"[INFO] Similarity result for image ID {image_id} updated successfully.")


async def send_image_to_inference(image_id: str):
    """
    Sends a single image to inference. Query's image bytes by image id from google storage and sends
    to torchserve.

    :param image_id: Unique image id used in google storage.
    :return: Result of inference, a list of features.
    """

    blob = bucket.blob(image_id)
    image_bytes = blob.download_as_bytes()

    # Asynchronous HTTP POST request to TorchServe using aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
                "http://127.0.0.1:8080/predictions/vitsimilaritymodel",
                data=image_bytes,
                headers={'Content-Type': 'application/octet-stream'}
        ) as response:
            result = await response.text()
            return result


def compute_cosine_similarity(old_image_id, new_result, existing_results: list) -> tuple:
    """
    Computes cosine similarity between list of lists of existing SQL results and new result list.

    :param new_result: New list inference result.
    :param existing_results: List of lists of previous inference results.
    :return: A float of cosine similarity.
    """
    max_similarity = -1
    most_similar_image_id = None
    for image_id, result in existing_results:
        if image_id != old_image_id:
            similarity = float(F.cosine_similarity(torch.tensor(new_result), torch.tensor(result)))
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_image_id = image_id
    return most_similar_image_id, max_similarity


async def process_images_and_store(new_files: list, connection):
    """
    Whole loop to process all new files, store inference results, and compute similarity.

    :param new_files: List of new uploaded files.
    """

    # Step 1: Process new files asynchronously
    new_files, results = await process_new_files(new_files)

    store_tasks = []
    loop = asyncio.get_event_loop()
    for image_id, result in zip(new_files, results):
        try:
            # Store inference result in the background
            store_task = loop.run_in_executor(None, store_inference_result, image_id, result, connection)
            store_tasks.append(store_task)
        except Exception as e:
            logging.error(f"[ERROR] Failed to store inference result for image {image_id}: {e}")

    # Wait for all store tasks to complete
    await asyncio.gather(*store_tasks)

    # Once all results are stored, move to similarity computation
    await compute_similarity_and_store(new_files, results, connection)


async def compute_similarity_and_store(new_files: list, results: list, connection):
    """
    Compute cosine similarity for the new images and store the results.

    :param new_files: List of newly uploaded images.
    :param results: List of inference results corresponding to the new images.
    :param connection: Database connection.
    """

    # Fetch existing results from the database for similarity computation
    existing_results = fetch_all_inference_results(connection)
    logging.info(type(existing_results))
    logging.info(type(new_files))
    logging.info(type(results))
    store_tasks = []
    loop = asyncio.get_event_loop()
    for image_id, result in zip(new_files, results):
        logging.info(f"TYPE OF RESULT IN RESULTS: {type(eval(result))}")
        try:
            most_similar_image_id, similarity = compute_cosine_similarity(image_id, eval(result),existing_results)
            logging.info(
                f"[INFO] Most similar image to {image_id} is {most_similar_image_id} with similarity {similarity}")

            # Store similarity result in the background
            store_task = loop.run_in_executor(None, store_similarity_result, image_id, most_similar_image_id,
                                              similarity, connection)
            store_tasks.append(store_task)
        except Exception as e:
            logging.exception(f"[ERROR] Failed to compute similarity for image {image_id}: {e}")

    # Wait for all store tasks to complete
    await asyncio.gather(*store_tasks)


def fetch_all_inference_results(pool):
    """
    Fetches all inference results from SQL database.
    :return: List of lists of existing results in SQL database.
    """

    select_stmt = sqlalchemy.text(
        "SELECT * FROM inference_results"
    )

    with pool.connect() as db_conn:
        # Execute the SELECT query to fetch all inference results
        result = db_conn.execute(select_stmt).fetchall()

    # Convert results into a list of tuples (image_id, inference_result)
    existing_results = [(row[0], eval(row[1])) for row in result]

    return existing_results


if __name__ == "__main__":
    loop = asyncio.get_event_loop()  # Use a single event loop
    engine = connect_with_connector()

    while True:
        try:
            new_images = check_new_upload(engine)

            if new_images:
                # Process new images and compute similarity
                loop.run_until_complete(process_images_and_store(new_files=new_images, connection=engine))

            time.sleep(5)  # Sleep for 5 seconds before checking again
        except Exception as e:
            logging.error(f"[ERROR] Critical exception occurred: {e}")
