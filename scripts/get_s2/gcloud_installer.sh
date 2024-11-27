#!/usr/bin/env bash

##########
# THIS SCRIPT IS TO PROGRAMMATICALLY DOWNLOAD AND INSTALL GCLOUD CLI
# I'LL PORT THIS LATER INTO A GD360 DOCKERFILE
##########

# download the archive silently and output to archive name
out_archive="$HOME/google-cloud-cli.tar.gz"
echo "downloading to $out_archive"
curl -sLo "$out_archive" https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz

# extract the archive
echo "extracting from ${out_archive} to ${HOME}"
tar -xf "$out_archive" -C "$HOME"

echo "running gcloud-sdk installer, log out and back into the instance for the changes to take effect"
$HOME/google-cloud-sdk/install.sh

###### NON INTERACTIVE DOESNT WORK YET :(

# run the installer in non-interactive mode ## HAVENT GOTTEN THIS WORKING YET # NEED TO DO SOMETHING WITH THE BASHRC TO ENABLE CLI
#CLOUDSDK_CORE_DISABLE_PROMPTS=1 $HOME/google-cloud-sdk/install.sh

# add gcloud to PATH in .bashrc
#echo -e "\n# The next line updates PATH for the Google Cloud SDK.\nif [ -f '\$HOME/google-cloud-sdk/path.bash.inc' ]; then . '\$HOME/google-cloud-sdk/path.bash.inc'; fi\n\n# The next line enables shell command completion for gcloud.\nif [ -f '\$HOME/google-cloud-sdk/completion.bash.inc' ]; then . '\$HOME/google-cloud-sdk/completion.bash.inc'; fi" >> ~/.bashrc

# reload .bashrc
#source ~/.bashrc