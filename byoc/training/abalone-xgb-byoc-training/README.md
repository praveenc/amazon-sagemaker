# BYO Training Image

## Notes

1. Dockerfile `ENTRYPOINT` should point to the `train.py` mode
1. Use generic `sagemaker.estimator.Estimator` to launch training job
1. To access env. variables from `train.py` use `from sagemaker_training import environment` module.
1. Ensure any additional files that `train.py` are also copied over to the container
1. Test locally using sagemaker LOCALMODE. see `local-train.py` for e.g.
1. Use `build_n_push.sh` shell script to build and push local docker image to ECR
1. 
