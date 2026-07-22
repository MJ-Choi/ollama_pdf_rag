echo "Check current chat.db"
sqlite3 web-ui/data/chat.db ".tables"

echo "Delete chat.db"
rm -f data/chat.db
rm -rf lib/db/migrations

echo "Generate DB"
pnpm db:generate

echo "Migrate DB"
pnpm db:migrate
