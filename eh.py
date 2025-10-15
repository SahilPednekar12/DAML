Search / web query snippets (what they try to do)

filetype:pdf r programming — Find PDF files about R programming.

intext: covid 19 vaccination details — Find pages that have the phrase “covid 19 vaccination details” in their text.

weather:india — Ask for weather information for India.

allintext: @copyright — Find pages that show “@copyright” in their content.

inurl:php intext:billgates — Find pages whose web address contains “php” and whose text mentions “billgates.”

video: hangover — Look for videos related to “hangover.”

info: www.telegram.com
 — Request summary/info about the site www.telegram.com
.

link: hello world — (old/rare) try to find pages that link to pages about “hello world.”

site: www.google.com
 — Limit a search to only pages on www.google.com
.

status: covid 19 patient — Not a standard search operator; probably trying to find status updates about a COVID-19 patient.

audio: hangover — Search for audio files (songs, recordings) named “hangover.”

movie: hangover — Search for movie info about “Hangover.”

book: harry potter — Search for books (or book info) called “Harry Potter.”

phonebooke: telegram — Looks like a typo; probably trying to search for contact info for Telegram.

list: movies — Ask for lists of movies.

Windows network commands (IP & DNS)

ipconfig — Show your computer’s IP address and basic network settings.

ipconfig /flushdns — Clear the computer’s stored (cached) DNS lookups so it asks fresh next time.

ipconfig /registerdns — Tell the PC to re-register its name and addresses with the network’s DNS server (refresh DNS registration).

netstat (connections & stats)

netstat — Show current network connections and open ports.

netstat -a — Show all connections plus any ports the computer is listening on.

netstat -s — Show counts and statistics for each network protocol (like TCP, UDP).

netstat -r — Show the routing table — where network traffic is sent next.

netstat -i — Show the network interfaces and basic packet counts (more common on Linux/macOS).

netstat -f — Show full domain names for the remote addresses instead of just IPs.

ping (test reachability)

ping www.google.com
 — Send small test messages to Google to check if it’s reachable and how long it takes.

ping -i s www.google.com
 — Invalid as written; likely intended to set the interval between pings (how often packets are sent).

ping -s 3 www.google.com
 — On Linux/macOS this sets packet size to number of bytes (here 3); on Windows that flag means something else or is invalid.

ping -c 1 www.google.com
 — Send only one ping packet (Linux/macOS).

ping -w 1 www.google.com
 — Set the timeout/deadline: wait only 1 (unit depends on OS) before giving up on a reply.

traceroute / tracert (show path to a server)

tracert — (Windows) Show each hop (router) between you and a destination so you see the path packets take.

tracert -h 2 www.google.com
 — Limit the traceroute to 2 hops (stop after 2 routers).

tracert -w 2 www.google.com
 — Wait 2 milliseconds for each reply before timing out (very short).

tracert -4 www.google.com
 — Force traceroute to use IPv4 addresses only.

nmap (port & host scanning)

Note: nmap is a network scanner — use it only on systems you own or are allowed to scan.

nmap -sS www.google.com
 — SYN (stealth) scan: quickly check which TCP ports are open.

nmap -sA www.google.com
 — ACK scan: test how a firewall treats packets (helps map firewall rules).

nmap -sF www.google.com
 — FIN scan: send FIN packets to try to detect open/closed ports in a stealthy way.

nmap -sN www.google.com
 — NULL scan: no TCP flags set — another stealthy port probe.

nmap -sX www.google.com
 — Xmas scan: set several TCP flags (FIN, PSH, URG) to probe ports stealthily.

 nmap -T4 -F www.google.com
— Fast/“quick” scan: -F scans fewer common ports (uses the ports list in nmap-services) and -T4 speeds up timing for quicker results.

nmap -T4 -A -v -p- www.google.com
— Intense/full scan: -p- scans all TCP ports, -A enables OS detection, version detection, script scanning and traceroute, -v increases verbosity, -T4 speeds timing.

nmap -sV --version-intensity 5 www.google.com
— Service/version detection: -sV probes open ports to determine service name and version; --version-intensity tunes how aggressive/version probes are (0–9).

nmap -sn 192.168.1.0/24
— Ping scan / host discovery: -sn (no port scan) just checks which hosts are up (ICMP / ARP / TCP pings), useful for live-host mapping. (Example uses a LAN range rather than google.)




Server Code

import socket

s= socket.socket()
print("Socket Created Successfully")

port = 8080
s.bind(("",port))
print("Socket is Listening")

s.listen(5)
print("Socket is Listening")

while True:
    c,addr = s.accept()
    print("Got connection from",addr)
    c.sendall('Thank you for connecting!'.encode())
    c.close


Client Code

import socket

s = socket.socket()
port = 8080

s.connect(('192.168.2.42',port))
print(s.recv(1024).decode())

