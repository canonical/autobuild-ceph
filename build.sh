#! /bin/bash
# Sourceable build-stage library.  Source this file, then call each stage

UBUNTU_BRANCH=${UBUNTU_BRANCH:-ubuntu/resolute}
DEBIAN_REF=${DEBIAN_REF:-origin/ubuntu/latest}
LAUNCHPAD_OWNER=${LAUNCHPAD_OWNER:-lmlogiudice}
CEPH_VERSION=${CEPH_VERSION:-20.2.0}

install_dependencies() {
    add_pbuilder_mirror
    sudo apt update
    sudo apt install -y devscripts git-buildpackage equivs python3-venv default-jdk javahelper dh-python
}

add_pbuilder_mirror() {
    # Fix 'Default mirror not found bug' - https://bugs-devel.debian.org/cgi-bin/bugreport.cgi?bug=1066021
    if grep -q '^MIRRORSITE=' /etc/pbuilderrc; then
        sudo sed -i 's|^MIRRORSITE=.*|MIRRORSITE=http://archive.ubuntu.com/ubuntu|' /etc/pbuilderrc
    else
        echo "MIRRORSITE=http://archive.ubuntu.com/ubuntu" | sudo tee -a /etc/pbuilderrc
    fi
}

prepare_tarball() {
    # check out upstream repo and generate tarball
    git clone https://github.com/ceph/ceph
    cd ceph
    git checkout "v${CEPH_VERSION}"
    ./make-dist
    cd ..

    mv ceph/ceph*.bz2 ceph-tarball.tar.bz2
    rm -rf ceph

    # check out launchpad repo
    git clone "git://git.launchpad.net/~${LAUNCHPAD_OWNER}/ubuntu/+source/ceph"
    cd ceph

    git remote add source git://git.launchpad.net/ubuntu/+source/ceph
    git checkout upstream && git checkout pristine-tar
    git fetch source
    git checkout -B "${UBUNTU_BRANCH}" "origin/${UBUNTU_BRANCH}"

    # incorporate tarball into a tag
    gbp import-orig --no-interactive --merge-mode=replace ../ceph-tarball.tar.bz2 -u "${CEPH_VERSION}"
    rm *.buildinfo || true
    git checkout "upstream/${CEPH_VERSION}"
    git checkout -b build
    git checkout ${DEBIAN_REF} -- debian
    git rm debian/compat || true
    git commit -m "add debian directory"
}

install_build_requirements() {
    sudo sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
    sudo mk-build-deps -i -t "apt-get -o Debug::pkgProblemResolver=1 -y --no-install-recommends" debian/control
    rm *.buildinfo *.changes *.deb || true
}

build() {
    debuild --no-lintian -us -uc -d
}

run_all() {
    install_dependencies
    add_pbuilder_mirror
    prepare_tarball
    install_build_requirements
    build
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -eux
    run_all
fi
