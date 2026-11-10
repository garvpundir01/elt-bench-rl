#!/usr/bin/env bash
# Build a local Terraform provider mirror for the ELT-Bench sandboxes (run once,
# on a network that can reach GitHub release downloads).
#
#   bash scripts/build_terraform_mirror.sh [mirror_dir]   # default ~/.cache/elt_rl/terraform-mirror
#
# Sandboxes mount the mirror read-only and install the pinned Airbyte provider
# (0.6.5) from it, so rollouts no longer download it on every `terraform init`.
set -euo pipefail
MIRROR=${1:-$HOME/.cache/elt_rl/terraform-mirror}
mkdir -p "$MIRROR"
cat > "$MIRROR/versions.tf" <<'EOF'
terraform {
  required_providers {
    airbyte = { source = "airbytehq/airbyte", version = "0.6.5" }
  }
}
EOF
docker run --rm -v "$MIRROR:/m" -w /m elt-swe terraform providers mirror -platform=linux_amd64 /m/providers
cat > "$MIRROR/terraformrc" <<'EOF'
provider_installation {
  filesystem_mirror {
    path    = "/opt/tf-mirror/providers"
    include = ["registry.terraform.io/airbytehq/airbyte"]
  }
  direct {
    exclude = ["registry.terraform.io/airbytehq/airbyte"]
  }
}
EOF
docker run --rm -v "$MIRROR:/m" alpine chown -R "$(id -u):$(id -g)" /m
find "$MIRROR/providers" -name '*.zip'
echo "mirror ready: $MIRROR"
