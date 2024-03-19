docker-compose -f deepseek.yml down --remove-orphans
docker-compose -f deepseek.yml build
docker-compose -f deepseek.yml up