#! /bin/bash

set -eux

sudo apt update
sudo apt install -y devscripts git-buildpackage equivs python3-venv default-jdk javahelper dh-python

# check out upstream repo and generate tarball
git clone https://github.com/ceph/ceph
cd ceph && ./make-dist && cd ..

mv ceph/ceph*.bz2 ceph-tarball.tar.bz2
rm -rf ceph

# check out launchpad repo
git clone git://git.launchpad.net/~lmlogiudice/ubuntu/+source/ceph
cd ceph

git remote add source git://git.launchpad.net/ubuntu/+source/ceph
git checkout upstream && git checkout pristine-tar
git fetch source
git checkout origin/ubuntu/resolute
git checkout -b ubuntu/resolute || true

# incorporate tarball into a tag
gbp import-orig --no-interactive --merge-mode=replace ../ceph-tarball.tar.bz2 -u 20.2.0
rm *.buildinfo || true
git checkout upstream/20.2.0
git checkout -b build
git checkout origin/ubuntu/latest -- debian
git commit -m "add debian directory"

sudo sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
sudo mk-build-deps -i -t "apt-get -o Debug::pkgProblemResolver=1 -y --no-install-recommends" debian/control
debuild --no-lintian -us -uc -d   # build the package
