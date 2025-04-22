import { StreamChat } from 'stream-chat';
import { v4 as uuid } from 'uuid';

// Simple in-memory store for invites
// In production, use a database
const invites: Record<string, { serverId: string, serverName: string, createdBy: string }> = {};

// Create a new invite
export async function POST(request: Request) {
  const body = await request.json();
  const { serverName, userId } = body;

  if (!serverName || !userId) {
    return Response.json({ error: 'Missing serverName or userId' }, { status: 400 });
  }

  const inviteCode = uuid().substring(0, 8);
  invites[inviteCode] = {
    serverId: serverName, // Using serverName as serverId since that's what the app uses
    serverName,
    createdBy: userId
  };

  return Response.json({ 
    inviteCode,
    inviteUrl: `${request.headers.get('origin')}/invite/${inviteCode}`
  });
}

// Get invite details
export async function GET(request: Request) {
  const url = new URL(request.url);
  const code = url.searchParams.get('code');

  if (!code || !invites[code]) {
    return Response.json({ error: 'Invalid invite code' }, { status: 404 });
  }

  return Response.json(invites[code]);
} 