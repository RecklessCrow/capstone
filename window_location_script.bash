# This script was found on https://unix.stackexchange.com/questions/14159/how-do-i-find-the-window-dimensions-and-position-accurately-including-decoration

# shellcheck disable=SC2046
eval $(xwininfo -name "Mesen - Galaga (J) [h1]" |
  sed -n -e "s/^ \+Absolute upper-left X: \+\([0-9]\+\).*/x=\1/p" \
         -e "s/^ \+Absolute upper-left Y: \+\([0-9]\+\).*/y=\1/p" \
         -e "s/^ \+Width: \+\([0-9]\+\).*/w=\1/p" \
         -e "s/^ \+Height: \+\([0-9]\+\).*/h=\1/p" )

echo -n "$w, $h, $x, $y"