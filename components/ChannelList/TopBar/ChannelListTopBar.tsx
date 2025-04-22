import { useState } from 'react';
import { ChevronDown, CloseIcon, PersonAdd } from '../Icons';
import ChannelListMenuRow from './ChannelListMenuRow';
import { menuItems } from './menuItems';
import { useDiscordContext } from '@/contexts/DiscordContext';
import { useChatContext } from 'stream-chat-react';
import { useClerk } from '@clerk/nextjs';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

export default function ChannelListTopBar({
  serverName,
}: {
  serverName: string;
}): JSX.Element {
  const [menuOpen, setMenuOpen] = useState(false);
  const [inviteUrl, setInviteUrl] = useState<string | null>(null);
  const [showInviteModal, setShowInviteModal] = useState(false);
  const { server } = useDiscordContext();
  const { client } = useChatContext();
  const { user } = useClerk();

  const handleMenuItemClick = async (itemName: string) => {
    setMenuOpen(false);
    
    if (itemName === 'Invite People') {
      await generateInviteLink();
      setShowInviteModal(true);
    }
  };

  const generateInviteLink = async () => {
    try {
      if (!server?.name || !user?.id) return;

      const response = await fetch('/api/invite', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          serverName: server.name,
          userId: user.id,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate invite link');
      }

      const data = await response.json();
      setInviteUrl(data.inviteUrl);
    } catch (error) {
      console.error('Error generating invite link:', error);
    }
  };

  const copyToClipboard = () => {
    if (!inviteUrl) return;
    
    navigator.clipboard.writeText(inviteUrl)
      .then(() => {
        toast.success('Invite link copied to clipboard!', {
          position: "top-center",
          autoClose: 3000,
        });
      })
      .catch(err => {
        console.error('Failed to copy:', err);
      });
  };

  return (
    <div className='w-full relative'>
      <ToastContainer />
      <button
        className={`flex w-full items-center justify-between p-4 border-b-2 ${
          menuOpen ? 'bg-gray-300' : ''
        } border-gray-300 hover:bg-gray-300`}
        onClick={() => setMenuOpen((currentValue) => !currentValue)}
      >
        <h2 className='text-lg font-bold text-gray-700'>{serverName}</h2>
        {menuOpen && <CloseIcon />}
        {!menuOpen && <ChevronDown />}
      </button>

      {menuOpen && (
        <div className='absolute w-full p-2 z-10'>
          <div className='w-full bg-white p-2 shadow-lg rounded-md'>
            {menuItems.map((option) => (
              <button
                key={option.name}
                className='w-full'
                onClick={() => handleMenuItemClick(option.name)}
              >
                <ChannelListMenuRow {...option} />
              </button>
            ))}
          </div>
        </div>
      )}

      {showInviteModal && (
        <div className='fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50'>
          <div className='bg-white rounded-lg p-6 max-w-md w-full'>
            <div className='flex justify-between items-center mb-4'>
              <h3 className='text-lg font-semibold text-gray-700'>Invite People to {serverName}</h3>
              <button 
                onClick={() => setShowInviteModal(false)}
                className='text-gray-500 hover:text-gray-700'
              >
                <CloseIcon />
              </button>
            </div>
            
            <div className='mb-4'>
              <p className='text-sm text-gray-600 mb-2'>Share this link with friends:</p>
              <div className='flex'>
                <input 
                  type='text'
                  readOnly
                  value={inviteUrl || ''}
                  className='flex-1 p-2 border border-gray-300 rounded-l-md bg-gray-50 text-sm'
                />
                <button
                  onClick={copyToClipboard}
                  className='bg-discord text-white px-4 py-2 rounded-r-md hover:bg-indigo-700'
                >
                  Copy
                </button>
              </div>
            </div>
            
            <div className='flex items-center mt-6'>
              <PersonAdd className='w-5 h-5 text-discord mr-2' />
              <span className='text-sm text-gray-600'>
                This invite link never expires
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
