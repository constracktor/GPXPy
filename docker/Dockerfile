FROM ubuntu:22.04

SHELL ["/bin/bash","-c"]

# install essentials
RUN apt-get update -y && apt-get install -y git wget xz-utils bzip2 libz-dev build-essential autoconf
RUN apt-get install -y vim cmake
# add user
RUN useradd -ms /bin/bash user
USER user
# create working directory
WORKDIR /home/user/

# clone repository
RUN git clone https://github.com/constracktor/hpx_project.git

# install dependencies
RUN cd hpx_project/dependencies && ./build-all.sh Release

RUN export APEX_SCREENcd_OUTPUT=1 && export APEX_CSV_OUTPUT=1

#cd && cd hpx_project/dependencies && git pull --rebase && ./build-all.sh Release
