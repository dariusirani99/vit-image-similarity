import cv2
import io
from google.cloud import storage
import random
import string
import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes
import pymysql
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# TODO: SET ABSOLUTE PATH OF YOUR GOOGLE APPLICATION CREDENTIALS HERE
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

# GLOBAL VARIABLES
# TODO: SET ABSOLUTE PATH OF YOUR GOOGLE APPLICATION CREDENTIALS HERE
client = storage.Client.from_service_account_json("")
camera = cv2.VideoCapture(0)  # '0' is the default camera index
# TODO: SET NAME OF YOUR GOOGLE BUCKET HERE
bucket_name = ""
bucket = client.bucket(bucket_name)


def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """

    # TODO: SET SQL INSTANCE PARAMETERS HERE
    instance_connection_name = ""
    db_user = ""
    db_pass = ""
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
    
engine = connect_with_connector()


# OTHER FUNCTIONS

def query_all_result(image_id: str, pool):
    """
    Query the inference result for a specific image_id from the SQL database.

    :param image_id: String of unique image id from the Google Cloud Storage.
    :param pool: The SQLAlchemy connection pool.
    :return: The inference result if found, or None if not found.
    """
    # Define the SELECT statement to query the inference result for a specific image_id
    select_stmt = sqlalchemy.text(
        """
        SELECT * FROM inference_results
        WHERE image_id = :image_id
        """
    )

    # Connect to the database and execute the SELECT query
    with pool.connect() as db_conn:
        result = db_conn.execute(select_stmt, {"image_id": image_id}).fetchone()

    if result:
        print(result)
        return result
    else:
        return None
        
        
def check_google_cloud_ids(bucket, image_id: str):
    """Checks if image id already exists, and returns True if it does"""
    blobs = bucket.list_blobs()
    for blob in blobs:
        if str(blob.name) == image_id:
            return True
    return False
  
  
def generate_image_id():
    length=16
    characters = string.ascii_letters + string.digits  # Includes both letters and digits
    random_id = ''.join(random.choice(characters) for _ in range(length))
    return random_id
    
  
def take_photo():
  # Check if the camera opened successfully
  if not camera.isOpened():
      print("Error: Could not open camera.")
  else:
      # Capture a single frame
      ret, frame = camera.read()
  
      # If frame was captured successfully
      if ret:
          # Encode the frame to a JPEG in-memory
          ret, buffer = cv2.imencode('.png', frame)
          
          # Convert to bytes
          image_bytes = io.BytesIO(buffer).getvalue()
          
          return image_bytes
      
      # Release the camera resource
      camera.release()
      

def upload_image_blob(bucket, image_stream, image_id: str):
  image_stream = io.BytesIO(image_bytes)
  image_stream.seek(0)
  
  blob = bucket.blob(blob_name=image_id)
  blob.upload_from_file(image_stream, content_type='image/png')


def display_image_from_gcs(bucket, image_id: str, similarity: float, location: str):
    """
    Downloads an image from Google Cloud Storage by image_id and displays it using matplotlib.

    :param bucket: The Google Cloud Storage bucket object.
    :param image_id: The unique image ID to fetch from the bucket.
    """
    try:
        # Get the blob from the bucket
        blob = bucket.blob(image_id)
        if not blob.exists():
            print(f"[ERROR] Image with ID '{image_id}' does not exist in the bucket.")
            return

        # Download the blob as bytes
        image_bytes = blob.download_as_bytes()
        
        # Convert bytes to numpy array and display using matplotlib
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is not None:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            title = f"Image ID: {image_id}\nSimilarity Score: {similarity}\nObject Location: {location}"
            plt.title(title, fontsize=10, loc='center')
            plt.show()
        else:
            print("[ERROR] Could not decode the image bytes.")
    except Exception as e:
        print(f"[ERROR] Failed to display the image. Error: {str(e)}")
        
        
  
if __name__ == "__main__":
  image_bytes = take_photo()
  if image_bytes is None:
    print("[ERROR] The image bytes are none. Exiting..")
    exit()
  while True:
    new_image_id = generate_image_id()
    duplicate_check = check_google_cloud_ids(bucket=bucket, image_id=new_image_id)
    if duplicate_check:
      continue
    else:
      break
  
  upload_image_blob(bucket = bucket, image_stream = image_bytes, image_id = new_image_id)
  time.sleep(5)
  
  # Getting result from VM
  result = (query_all_result(image_id = new_image_id, pool=engine))
  result_old = (query_all_result(image_id = result[4], pool=engine))
  
  # displaying to user
  if result[4]:
    display_image_from_gcs(bucket=bucket, image_id=result[4], similarity=result[5], location=result_old[3])
  
  if result:
    print(f"The most similar image ID is: {result[4]}.\nThis object is located at: {result_old[3]}.\nThis object has a similarity score of: {result[5]}\n\n")
  else:
    print("No similar result found...\n\n")
  