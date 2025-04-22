'use client';

import { useParams, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { useClerk } from '@clerk/nextjs';
import { useDiscordContext } from '@/contexts/DiscordContext';
import { useChatContext } from 'stream-chat-react';
import { useStreamVideoClient } from '@stream-io/video-react-sdk';
import { LoadingIndicator } from 'stream-chat-react';

export default function InvitePage() {
  const params = useParams();
  const router = useRouter();
  const { user } = useClerk();
  const { createServer } = useDiscordContext();
  const { client } = useChatContext();
  const videoClient = useStreamVideoClient();
  
  const [isLoading, setIsLoading] = useState(true);
  const [serverDetails, setServerDetails] = useState<{
    serverId: string;
    serverName: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchInviteDetails = async () => {
      try {
        if (!params.code) {
          setError('Invalid invite link');
          setIsLoading(false);
          return;
        }

        const response = await fetch(`/api/invite?code=${params.code}`);
        
        if (!response.ok) {
          setError('This invite is invalid or has expired');
          setIsLoading(false);
          return;
        }

        const data = await response.json();
        setServerDetails({
          serverId: data.serverId,
          serverName: data.serverName,
        });
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching invite details:', error);
        setError('Failed to load invitation details');
        setIsLoading(false);
      }
    };

    fetchInviteDetails();
  }, [params.code]);

  const handleJoinServer = async () => {
    try {
      if (!client || !videoClient || !user || !serverDetails) {
        setError('Unable to join server at this time');
        return;
      }

      // Join the server by creating it for the current user
      // Since we don't have a real "join" function, we're using createServer
      // with the same server name, which will connect the user to the existing server
      await createServer(
        client,
        videoClient,
        serverDetails.serverName,
        `https://getstream.io/random_png/?id=${serverDetails.serverName}`,
        [user.id]
      );

      // Redirect to home page
      router.push('/');
    } catch (error) {
      console.error('Error joining server:', error);
      setError('Failed to join server');
    }
  };

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-800">
        <LoadingIndicator />
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen flex flex-col items-center justify-center bg-gray-800 text-white p-4">
        <h1 className="text-2xl font-bold mb-4">Invite Error</h1>
        <p className="mb-6">{error}</p>
        <button
          onClick={() => router.push('/')}
          className="bg-discord hover:bg-indigo-700 text-white font-medium py-2 px-4 rounded"
        >
          Return Home
        </button>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-gray-800 text-white p-4">
      <div className="bg-gray-700 p-8 rounded-lg max-w-md w-full">
        <h1 className="text-2xl font-bold mb-4">Server Invitation</h1>
        <p className="mb-6">
          You've been invited to join <span className="font-semibold">{serverDetails?.serverName}</span>
        </p>
        
        <div className="flex flex-col space-y-4">
          <button
            onClick={handleJoinServer}
            className="bg-discord hover:bg-indigo-700 text-white font-medium py-2 px-4 rounded"
          >
            Accept Invite
          </button>
          
          <button
            onClick={() => router.push('/')}
            className="bg-gray-600 hover:bg-gray-500 text-white font-medium py-2 px-4 rounded"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
} 