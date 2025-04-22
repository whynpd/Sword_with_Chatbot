import {
  Emoji,
  GIF,
  PlusCircle,
  Present,
  FolderPlus,
  Chart,
} from '@/components/ChannelList/Icons';
import { useEffect, useRef, useState } from 'react';
import { Message, SendButton, useChatContext } from 'stream-chat-react';
import { plusItems } from './plusItems';
import ChannelListMenuRow from '@/components/ChannelList/TopBar/ChannelListMenuRow';
import { analyzeBatchSentiment, SentimentResponse } from '@/services/sentimentService';
import SentimentDisplay from '../SentimentAnalysis/SentimentDisplay';

export default function MessageComposer(): JSX.Element {
  const [plusMenuOpen, setPlusMenuOpen] = useState(false);
  const { channel } = useChatContext();
  const [message, setMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showSentiment, setShowSentiment] = useState(false);
  const [sentimentAnalysis, setSentimentAnalysis] = useState<SentimentResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recentMessages, setRecentMessages] = useState<string[]>([]);

  // Load recent messages when channel changes
  useEffect(() => {
    if (channel) {
      const loadRecentMessages = async () => {
        try {
          const response = await channel.query({
            messages: { limit: 100 }
          });
          
          const messages = response.messages || [];
          const messageTexts = messages
            .filter((msg) => msg.text && msg.text.trim() !== '')
            .map((msg) => msg.text as string);
            
          setRecentMessages(messageTexts);
        } catch (error) {
          console.error('Error loading recent messages:', error);
        }
      };
      
      loadRecentMessages();
    }
  }, [channel]);

  // Function to analyze sentiment of a message
  const analyzeMessageSentiment = async (messageText: string) => {
    setIsAnalyzing(true);
    setShowSentiment(true);
    
    try {
      if (recentMessages.length > 0) {
        // Use the recent message history plus the current message
        const allMessages = [...recentMessages, messageText];
        const analysis = await analyzeBatchSentiment(allMessages);
        setSentimentAnalysis(analysis);
      } else {
        // Fallback if no message history
        setSentimentAnalysis({
          message: messageText,
          sentiment: 'neutral',
          confidence: 1,
          probabilities: {
            positive: 0.33,
            neutral: 0.34,
            negative: 0.33
          },
          recommendation: "No historical messages available for baseline comparison."
        });
      }
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
      setSentimentAnalysis(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !channel) return;

    try {
      const response = await channel.sendFile(file);
      const fileUrl = response.file;
      
      // Send a message with the file attachment
      await channel.sendMessage({
        text: message || file.name,
        attachments: [
          {
            type: 'file',
            asset_url: fileUrl,
            title: file.name,
            mime_type: file.type,
            file_size: file.size,
          },
        ],
      });
      
      // Add message to recent messages
      if (message) {
        setRecentMessages([...recentMessages, message]);
      }
      
      setMessage('');
      setPlusMenuOpen(false);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleUploadClick = (option: any) => {
    if (option.name === 'Upload a File') {
      fileInputRef.current?.click();
    }
    setPlusMenuOpen(false);
  };

  const handleSendMessage = () => {
    if (!message.trim() || !channel) return;
    
    channel.sendMessage({ text: message });
    
    // Add the sent message to our recent messages
    setRecentMessages([...recentMessages, message]);
    
    setMessage('');
  };

  return (
    <div className='flex flex-col mx-6 my-3'>
      {showSentiment && sentimentAnalysis && (
        <SentimentDisplay analysis={sentimentAnalysis} isLoading={isAnalyzing} />
      )}
      
      <div className='flex px-4 py-1 bg-composer-gray items-center justify-center space-x-4 rounded-md text-gray-600 relative'>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileUpload}
          className="hidden"
        />
        <button onClick={() => setPlusMenuOpen((menuOpen) => !menuOpen)}>
          <PlusCircle className='w-8 h-8 hover:text-gray-800' />
        </button>
        {plusMenuOpen && (
          <div className='absolute p-2 z-10 -left-6 bottom-12'>
            <div className='bg-white p-2 shadow-lg rounded-md w-40 flex flex-col'>
              {plusItems.map((option) => (
                <button
                  key={option.name}
                  className=''
                  onClick={() => handleUploadClick(option)}
                >
                  <ChannelListMenuRow {...option} />
                </button>
              ))}
            </div>
          </div>
        )}
        <input
          className='flex-grow border-transparent bg-transparent outline-none text-sm font-semibold m-0 text-gray-normal'
          type='text'
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder='Message #general'
        />
        
        <button 
          onClick={() => analyzeMessageSentiment(message)}
          className='w-8 h-8 hover:text-gray-800'
          title="Analyze sentiment"
          disabled={!message.trim()}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </button>
        
        <Present className='w-8 h-8 hover:text-gray-800' />
        <GIF className='w-8 h-8 hover:text-gray-800' />
        <Emoji className='w-8 h-8 hover:text-gray-800' />
        <SendButton
          sendMessage={handleSendMessage}
        />
      </div>
    </div>
  );
}
