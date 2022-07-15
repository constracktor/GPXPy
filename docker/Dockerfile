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
RUN apt-get update -y && apt-get install -y vim make cmake git python3 gcc g++ openmpi-bin xz-utils pip sudo

RUN pip install --user clingo

# add user
RUN useradd -ms /bin/bash user  && echo "user:user" | chpasswd && adduser user sudo
USER user
# create working directory
WORKDIR /home/user/




# clone repository
RUN git clone https://github.com/constracktor/hpx_project.git

# install spack
RUN git clone -c feature.manyFiles=true https://github.com/spack/spack.git
# configure spack
RUN spack/bin/spack compiler find && spack/bin/spack external find python
# install hpx
#RUN spack/bin/spack install hpx

#spack/bin/spack compiler find
#spack/bin/spack external find python
#pip install --user clingo
#/home/user/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.2.0/hpx-1.8.0-rrzfvta5vlvp2r62ysqnnkjnfwbjal7d





# remove docker error message (somehow does not work and needs to be done manually)
RUN export OMPI_MCA_btl_vader_single_copy_mechanism=none
