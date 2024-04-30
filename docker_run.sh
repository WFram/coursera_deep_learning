docker run \
    --rm \
    --name coursera_deep_learning \
    -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e HOME=$HOME \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -w $(pwd) \
    -v $HOME:$HOME \
    -v /media:/media \
    --device=/dev/dri:/dev/dri \
    -it \
    coursera_deep_learning:main \
    bash