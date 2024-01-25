#!/bin/bash

GPUSER="mds"
GPU_HOSTNAME="10.222.102.57"
GPU_HOSTNAME_VIA_PROXY="10.3.5.7"

# GPUSER=$USER
# GPU_HOSTNAME=$HOSTNAME
# GPU_HOSTNAME_VIA_PROXY=$HOSTNAME
LOG_FILE=$(basename $0 | cut -d "." -f1).log

# Fonction pour nettoyer les ressources et terminer les processus
cleanup() {
    #echo "Nettoyage en cours..."
    
    # Tuer les processus SSH
    pkill -P $$
    exit 0
}

# Définir le trap pour nettoyer les ressources en cas d'interruption du script
trap 'cleanup' SIGINT SIGTERM

# Ouverture tunnel un
tunnel_one="ssh -p 22 $GPUSER@$GPU_HOSTNAME_VIA_PROXY 'jupyter-notebook --no-browser 2>&1 &' > $LOG_FILE 2>&1 &"
eval $tunnel_one

sleep 5

# Recuperation de l'url depuis le fichier de log
url=$(grep -E  "^([[:space:]]){1,}http://localhost:[0-9]{4}/*" $LOG_FILE)
port=$(echo "$url" | grep -oP 'localhost:\K[0-9]+')

echo URL: $url
echo Port: $port

tunnel_two="ssh -N -L $port:localhost:8888 $GPUSER@$GPU_HOSTNAME &"
eval $tunnel_two

if [[ -f "$LOG_FILE" ]]; then rm $LOG_FILE; fi

# Lancement de l'url dans le navigateur
firefox "$url" &

# Attente des 3 process: 2 ssh et 1 onglet firefox
wait