.DELETE_ON_ERROR:

AWS_REGION := us-east-1
ECR_REPOSITORY := my-repo
DOCKER_IMAGE := abalone-train

# Delete all files under the models directory
clean:
	rm -rf models/*

# Build and push the Docker image to ECR
push-image:
	./build_n_push.sh