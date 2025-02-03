#!/bin/bash
bindir='/usr/bin'
D='/build'
includedir='/usr/include'
libdir='/usr/lib'
TARGET_ARCH='aarch64'
WORKDIR='/workdir'
BPN=${PN}
sysconfdir='/etc'
mkdir -p ${D}
mkdir -p ${WORKDIR}
GCC_ARCH="aarch64-linux-gnu"
# SRCBRANCH='lf-6.6.36_2.1.0'
SRCBRANCH='lf-5.15.71_2.2.2'
