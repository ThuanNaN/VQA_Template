# Image features extracted from Up-Down Attention with 36 objects

## Download using MinIO

This directory contains a script to download object features from a MinIO storage bucket.

### Prerequisites

1. Install the required package:

```bash
pip install minio
```

2. Configure MinIO credentials in your `.env` file:

```bash
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET=vqa-features
MINIO_SECURE=False  # Set to True if using HTTPS
```

### Usage

Run the download script:

```bash
python download.py
```

The script provides multiple options:

1. **Download all .tsv files** - Downloads all TSV feature files from the bucket
2. **Download all objects** - Downloads everything from the bucket
3. **Download with prefix** - Downloads objects matching a specific prefix (e.g., "train/", "val/")
4. **Download specific object** - Downloads a single file by name

All downloaded files will be saved to this directory (`obj36_feat/`).

### Manual Download

Alternatively, download all *.tsv files to this folder manually.
