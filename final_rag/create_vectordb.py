# Run this once before running the streamlit app to create the vector database
# Ensure you have Weaviate running in Docker before executing this script
import os
import weaviate
import weaviate.classes as wvc
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
import pickle

PDF_DIRECTORY = "articles"
COLLECTION_NAME = "PDFChunksBigger"
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'


def process_pdfs_for_chunks(pdf_dir):
    all_chunks = []
    if not os.path.exists(pdf_dir) or not os.listdir(pdf_dir):
        print(
            f"Directory '{pdf_dir}' is empty or does not exist. Please add your PDFs.")
        return all_chunks

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF(s) to process...")

    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        try:
            elements = partition_pdf(
                filename=filepath, strategy="hi_res",
                infer_table_structure=True, chunking_strategy="by_title"
            )
            for chunk in elements:
                chunk.metadata.filename = filename
            all_chunks.extend(elements)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return all_chunks


if __name__ == "__main__":
    # Connecting to Weaviate
    try:
        client = weaviate.connect_to_local(skip_init_checks=True)
        print("Successfully connected to Weaviate.")
    except Exception as e:
        print(
            f"Could not connect to Weaviate. Please ensure it's running in Docker. Error: {e}")
        exit()

    # Delete the collection if already present
    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Deleting it.")
        client.collections.delete(COLLECTION_NAME)

    # Process PDFs
    if (os.path.exists("chunks_cache.pkl")):
        # Load chunks from cache if available
        print("Loading chunks from cache...")
        with open("chunks_cache.pkl", "rb") as f:
            final_chunks = pickle.load(f)
        print("Successfully loaded chunks from cache.")
    else:
        # Process PDFs to create chunks if cache is not available
        print("No cached chunks found. Processing PDFs...")
        final_chunks = process_pdfs_for_chunks(PDF_DIRECTORY)
    if not final_chunks:
        print("No chunks were created. Exiting.")
        exit()

    # Generate Embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts_to_embed = [chunk.text for chunk in final_chunks]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    # Create Weaviate Collection with explicit vector config of HNSW
    print(f"Creating collection: {COLLECTION_NAME}...")
    chunks_collection = client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            wvc.config.Property(
                name="text", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(
                name="filename", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(name="page_number",
                                data_type=wvc.config.DataType.INT),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE
        )
    )

    # Batch Import Data into Weaviate
    print(f"Importing {len(final_chunks)} chunks into Weaviate...")
    with chunks_collection.batch.dynamic() as batch:
        for i, chunk in enumerate(final_chunks):
            properties = {
                "text": chunk.text,
                "filename": chunk.metadata.filename,
                "page_number": chunk.metadata.page_number,
            }
            batch.add_object(
                properties=properties,
                vector=embeddings[i].tolist()
            )
            if (i + 1) % 500 == 0:
                print(f"-> Indexed {i + 1}/{len(final_chunks)} chunks")

    print("\n✅ Text-only database build complete!")
    client.close()
