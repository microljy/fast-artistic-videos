set -e
# Get a carriage return into `cr`
cr=`echo $'\n.'`
cr=${cr%.}


# Find out whether ffmpeg or avconv is installed on the system
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
  FFMPEG=avconv
  command -v $FFMPEG >/dev/null 2>&1 || {
    echo >&2 "This script requires either ffmpeg or avconv installed.  Aborting."; exit 1;
  }
}

if [ "$#" -le 1 ]; then
   echo "Usage: ./stylizeVideo <path_to_video> <path_to_style_image>"
   exit 1
fi

# Parse arguments
filename=$(basename "$1")
extension="${filename##*.}"
filename="${filename%.*}"
filename=${filename//[%]/x}
style_image=$2

# Create output folder
mkdir -p $filename



# Save frames of the video as individual image files
if [ -z $resolution ]; then
  $FFMPEG -i $1 ${filename}/frame_%04d.ppm
  resolution=default
else
  $FFMPEG -i $1 -vf scale=$resolution ${filename}/frame_%04d.ppm
fi


# Create video from output images.
$FFMPEG -i ${filename}/out-%04d.png ${filename}-stylized.$extension
