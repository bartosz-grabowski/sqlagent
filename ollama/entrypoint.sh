#!/bin/bash

/bin/ollama serve &
pid=$!
sleep 5
echo "❗ Downloading model: $OLLAMA_MODEL"
ollama pull $OLLAMA_MODEL
echo "✅ Done downloading model"
wait $pid
