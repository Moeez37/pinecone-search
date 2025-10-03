#!/bin/bash

# Script to help with Docker operations

case "$1" in
  build)
    echo "Building Docker image..."
    docker-compose build
    ;;
  up)
    echo "Starting containers..."
    docker-compose up
    ;;
  up-d)
    echo "Starting containers in detached mode..."
    docker-compose up -d
    ;;
  down)
    echo "Stopping containers..."
    docker-compose down
    ;;
  logs)
    echo "Showing logs..."
    docker-compose logs -f
    ;;
  restart)
    echo "Restarting containers..."
    docker-compose restart
    ;;
  *)
    echo "Usage: $0 {build|up|up-d|down|logs|restart}"
    exit 1
    ;;
esac

exit 0