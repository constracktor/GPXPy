# HOW TO
# Build image:
# sudo docker build . -f docker/Dockerfile -t hpx_image
#
# Run container inside petsc_project folder:
# sudo docker run -it --rm --user user --name hpx_container hpx_image
#
# Compile Code:
# cd hpx_project/src &&
#
# Run Code:
#
#
# Update Repository:
# git add . && git commit -m "comment" && git push
# git add . && git commit --amend --no-edit && git push -f

FROM ubuntu:22.04

SHELL ["/bin/bash","-c"]
# install essentials
RUN apt-get update -y && apt-get install -y vim make cmake git gcc g++ openmpi-bin spack

# add user
RUN useradd -ms /bin/bash user
USER user
# create working directory
WORKDIR /home/user/
# clone repository
RUN git clone https://github.com/constracktor/hpx_project.git

RUN spack install hpx





# remove docker error message (somehow does not work and needs to be done manually)
RUN export OMPI_MCA_btl_vader_single_copy_mechanism=none
