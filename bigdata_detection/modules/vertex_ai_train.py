from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project='impactful-arbor-438008-r7', location='us-central1')

# Create and configure the custom training job
job = aiplatform.CustomJob.from_local_script(
    display_name='my_training_job',
    script_path='gs://space_things/multiSVM.py',  # Path to your updated training script in GCS
    container_uri='us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-24:latest',  # Use appropriate container
    requirements=['pandas', 'scikit-learn', 'joblib', 'numpy'],  # Add other dependencies
    args=[
        '--not_storm_path', 'gs://space_things/NOT_STORM/',
        '--storm_labelled_path', 'gs://space_things/STORM_LABELLED/',
        '--model_output', 'gs://space_things/model.joblib'
    ],
    staging_bucket='gs://space_things',  # Bucket for staging
    replica_count=1,
    machine_type='n1-standard-4',  # Modify based on your needs
)

# Run the job
try:
    job.run(sync=False)  # Set sync=True if you want to wait for the job to complete
except Exception as e:
    print(f"Error occurred: {e}")
