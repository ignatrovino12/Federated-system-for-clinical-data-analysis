Usage (development / self-signed):

1. Generate self-signed certs:

   cd federated_server/docker/nginx
   chmod +x generate-self-signed.sh
   ./generate-self-signed.sh

2. Access Flower via HTTPS on port 443 of the host. For self-signed certs you may need to accept the certificate in the browser or use `curl -k https://localhost/`.

Production notes:
- Replace the self-signed certs in `certs/` with real certificates from Let's Encrypt or your CA.
- Optionally use the `nginx` image with certbot sidecar for automatic issuance.
 
- The compose setup mounts the same host `./nginx/certs` into both nginx and certbot so copying the live certs to the flat files `fullchain.pem` / `privkey.pem` makes the nginx config work without needing template substitution.
- future automated renewer service that runs `certbot renew` periodically and reloads nginx on success.
