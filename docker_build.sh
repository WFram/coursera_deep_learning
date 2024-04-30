docker image build \
    -t \
    coursera_deep_learning:main \
    --build-arg \
    USER_ID=$(id -u) \
    --build-arg \
    GROUP_ID=$(id -g) \
    -f \
    ./Dockerfile \
    .