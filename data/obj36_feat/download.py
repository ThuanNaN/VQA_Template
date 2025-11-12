import os
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
load_dotenv()

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "vqa-features")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
    raise ValueError("MinIO credentials are not set in environment variables.")


def get_minio_client():
    """Initialize and return MinIO client."""
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    return client


def download_object(client, object_name, download_path):
    """Download a single object from MinIO bucket."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        
        # Download the object
        client.fget_object(MINIO_BUCKET, object_name, download_path)
        print(f"Downloaded: {object_name} -> {download_path}")
        return True
    except S3Error as e:
        print(f"Error downloading {object_name}: {e}")
        return False


def list_objects(client, prefix=""):
    """List all objects in the MinIO bucket with optional prefix."""
    try:
        objects = client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
        object_list = [obj.object_name for obj in objects]
        return object_list
    except S3Error as e:
        print(f"Error listing objects: {e}")
        return []


def download_all_objects(client, prefix="", download_dir="./"):
    """Download all objects from MinIO bucket with optional prefix filter."""
    objects = list_objects(client, prefix)
    
    if not objects:
        print(f"No objects found with prefix: {prefix}")
        return
    
    print(f"Found {len(objects)} objects to download")
    
    success_count = 0
    for obj_name in objects:
        # Create local path maintaining the object structure
        local_path = os.path.join(download_dir, obj_name)
        
        if download_object(client, obj_name, local_path):
            success_count += 1
    
    print(f"\nDownload completed: {success_count}/{len(objects)} objects downloaded successfully")


def download_tsv_files(client, download_dir="./"):
    """Download all .tsv files from MinIO bucket."""
    objects = list_objects(client)
    tsv_objects = [obj for obj in objects if obj.endswith('.tsv')]
    
    if not tsv_objects:
        print("No .tsv files found in the bucket")
        return
    
    print(f"Found {len(tsv_objects)} .tsv files to download")
    
    success_count = 0
    for obj_name in tsv_objects:
        # Download to current directory, keeping only filename
        filename = os.path.basename(obj_name)
        local_path = os.path.join(download_dir, filename)
        
        if download_object(client, obj_name, local_path):
            success_count += 1
    
    print(f"\nDownload completed: {success_count}/{len(tsv_objects)} .tsv files downloaded successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("MinIO Object Downloader for VQA Features")
    print("=" * 60)
    print(f"Endpoint: {MINIO_ENDPOINT}")
    print(f"Bucket: {MINIO_BUCKET}")
    print(f"Secure: {MINIO_SECURE}")
    print("=" * 60)
    
    # Initialize MinIO client
    client = get_minio_client()
    
    # Check if bucket exists
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            print(f"Error: Bucket '{MINIO_BUCKET}' does not exist")
            exit(1)
    except S3Error as e:
        print(f"Error checking bucket: {e}")
        exit(1)
    
    # Get current directory (obj36_feat)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\nDownload Options:")
    print("1. Download all .tsv files")
    print("2. Download all objects from bucket")
    print("3. Download objects with specific prefix")
    print("4. Download specific object")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        download_tsv_files(client, current_dir)
    elif choice == "2":
        download_all_objects(client, download_dir=current_dir)
    elif choice == "3":
        prefix = input("Enter prefix (e.g., 'features/' or 'train/'): ").strip()
        download_all_objects(client, prefix=prefix, download_dir=current_dir)
    elif choice == "4":
        object_name = input("Enter object name: ").strip()
        local_path = os.path.join(current_dir, os.path.basename(object_name))
        download_object(client, object_name, local_path)
    else:
        print("Invalid choice")
