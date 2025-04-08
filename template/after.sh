echo "Waiting for the Whisper UI server to open port ${SERVER_PORT}..."

# Check if the server is listening on host:port within a 120-second timeout.
if wait_until_port_used "${host}:${SERVER_PORT}" 120; then
  echo "Server is listening on port ${SERVER_PORT}!"
else
  echo "Timed out waiting for server on port ${SERVER_PORT}."
  # The following lines stop the job if the port never opens:
  pkill -P ${SCRIPT_PID}
  clean_up 1
fi

# Add a brief delay to allow everything to settle
sleep 2
