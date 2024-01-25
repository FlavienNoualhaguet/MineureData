#!/bin/bash

# GPUSER="mds"
# GPU_HOSTNAME="10.222.102.57"
# GPU_HOSTNAME_VIA_PROXY="10.3.5.7"
# GPU_PORT=3014

GPUSER=$USER
GPU_HOSTNAME=$HOSTNAME
GPU_HOSTNAME_VIA_PROXY=$HOSTNAME
GPU_PORT=22

LOGFILE=$(basename $0 | cut -d "." -f1).log

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
ssh -p $GPU_PORT $GPUSER@$GPU_HOSTNAME_VIA_PROXY -o ConnectTimeout=5 'jupyter-notebook --no-browser 2>&1 &' > $LOGFILE 2>&1 &
sleep 5

# Recuperation de l'url depuis le fichier de log
url=$(grep -E  "^([[:space:]]){1,}http://localhost:[0-9]{4}/*" $LOGFILE)
port=$(echo "$url" | grep -oP 'localhost:\K[0-9]+')

if [[ -z "$url" ]]; then
    printf "Pas d'url trouvee pour jupyter-notebook.\n"
    printf "Il y a surement un probleme de connexion ssh\n"
    exit 1
fi

printf "Jupyter-Notebook URL: $url\n"
printf "Port: $port\n"

# Ouverture tunnel deux
ssh -N -L $port:localhost:8888 $GPUSER@$GPU_HOSTNAME &

if [[ -f "$LOGFILE" ]]; then rm $LOGFILE; fi

# Lancement de l'url dans le navigateur
firefox "$url" &

# Attente des 3 process: 2 ssh et 1 onglet firefox
wait