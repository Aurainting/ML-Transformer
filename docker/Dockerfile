FROM archlinux:latest

# Labels
LABEL author="aurumting"
LABEL version="1.0"
LABEL description="mlpack implementation of Transformer"

# Mirrors in China
RUN sed -i '1iServer = https://mirrors.tuna.tsinghua.edu.cn/archlinux/$repo/os/$arch' /etc/pacman.d/mirrorlist \
    && sed -i '1iServer = https://mirrors.ustc.edu.cn/archlinux/$repo/os/$arch' /etc/pacman.d/mirrorlist \
    && sed -i '1iServer = https://mirrors.aliyun.com/archlinux/$repo/os/$arch' /etc/pacman.d/mirrorlist \
    && sed -i '1iServer = http://mirrors.aliyun.com/archlinux/$repo/os/$arch' /etc/pacman.d/mirrorlist \
    && sed -i '1iServer = http://mirrors.163.com/archlinux/$repo/os/$arch' /etc/pacman.d/mirrorlist

# https://hub.docker.com/_/archlinux
# Create an lsign-key and be careful to not redistribute that key
# Then download essential packages
RUN pacman-key --init && pacman -Syyu --noconfirm && pacman -S --noconfirm vim git sudo base-devel gcc gdb cmake ninja

# Create new user, delete the password, and allow passwordless sudo
RUN useradd -m aurumting && passwd -d aurumting \
    && printf 'aurumting ALL=(ALL) ALL\n' | tee -a /etc/sudoers

WORKDIR /home/aurumting
# Install Yay
RUN sudo -u aurumting bash -c 'git clone https://aur.archlinux.org/yay.git && cd yay && makepkg -si --noconfirm'

# Build mlpack
RUN sudo -u aurumting bash -c 'yay -Y --gendb && yay -Syu --devel && yay -Y --devel --save && yay -S mlpack --noconfirm'

# Add source files
# You should in the source directory and use `-f` to specify this Dockerfile
COPY . /home/aurumting/ml_transformer
