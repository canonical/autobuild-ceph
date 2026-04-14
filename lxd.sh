lxd_cmd() {
    sudo -g lxd lxc "$@"
}

lxd_launch_and_wait() {
    local image="$1"
    local container="$2"
    lxd_cmd launch "${image}" "${container}"
    lxd_cmd exec "${container}" -- cloud-init status --wait
}

lxd_push_file() {
    local source_path="$1"
    local container="$2"
    local target_path="$3"
    lxd_cmd file push "${source_path}" "${container}/${target_path}"
}

lxd_exec_build_stage() {
    local container="$1"
    local stage_cmd="$2"
    lxd_cmd exec "${container}" \
        --env UBUNTU_BRANCH="${UBUNTU_BRANCH}" \
        --env DEBIAN_REF="${DEBIAN_REF}" \
        --env LAUNCHPAD_OWNER="${LAUNCHPAD_OWNER}" \
        --env CEPH_VERSION="${CEPH_VERSION}" \
        -- bash -c "${stage_cmd}"
}
