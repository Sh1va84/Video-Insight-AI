import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase'
import { createSupabaseClient } from '../helpers/supabaseClient.js'
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
import { YoutubeTranscript } from 'youtube-transcript'

// Helper function to extract video ID from YouTube URL
function extractVideoId(url) {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
    /^([a-zA-Z0-9_-]{11})$/
  ]
  
  for (const pattern of patterns) {
    const match = url.match(pattern)
    if (match) return match[1]
  }
  return null
}

// Helper function to get video title using oEmbed API (no API key needed)
async function getVideoTitle(videoId) {
  try {
    const response = await fetch(
      `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`
    )
    if (response.ok) {
      const data = await response.json()
      return data.title
    }
  } catch (error) {
    console.log('Could not fetch video title:', error.message)
  }
  return 'Unknown Title'
}

export async function storeDocument(req) {
  try {
    if (!req?.body?.url) {
      throw new Error('URL is required in the request body')
    }

    const { url, documentId } = req.body
    console.log(`📥 Processing document: ${url}`)

    // Extract video ID
    const videoId = extractVideoId(url)
    if (!videoId) {
      throw new Error('Invalid YouTube URL. Could not extract video ID.')
    }

    console.log(`🎬 Video ID: ${videoId}`)

    const supabase = createSupabaseClient()

    // Verify API key is available
    if (!process.env.GEMINI_API_KEY) {
      throw new Error('GEMINI_API_KEY is not configured in environment variables')
    }

    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: 'embedding-001',
      apiKey: process.env.GEMINI_API_KEY
    })

    const vectorStore = new SupabaseVectorStore(embeddings, {
      client: supabase,
      tableName: 'embedded_documents',
      queryName: 'match_documents'
    })

    console.log('🎥 Fetching YouTube transcript...')

    let transcript
    try {
      transcript = await YoutubeTranscript.fetchTranscript(videoId)
    } catch (transcriptError) {
      console.error('Transcript Error:', transcriptError.message)
      
      if (transcriptError.message.includes('disabled') || 
          transcriptError.message.includes('Transcript is disabled')) {
        throw new Error('Transcripts are disabled for this video. Please try a video with captions enabled.')
      }
      
      if (transcriptError.message.includes('not found') ||
          transcriptError.message.includes('No transcript')) {
        throw new Error('No transcript found for this video. Please try a video with captions/subtitles.')
      }

      throw new Error(`Failed to fetch transcript: ${transcriptError.message}`)
    }

    if (!transcript || transcript.length === 0) {
      throw new Error('No transcript content found for this video.')
    }

    // Combine transcript segments into full text
    const fullTranscript = transcript.map(segment => segment.text).join(' ')
    
    // Get video title
    const videoTitle = await getVideoTitle(videoId)
    console.log(`📝 Video Title: "${videoTitle}"`)

    // Create document content with title
    const pageContent = `Video title: ${videoTitle} | Video context: ${fullTranscript}`

    console.log('✂️ Splitting document into chunks...')
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200
    })

    const texts = await textSplitter.createDocuments([pageContent])

    if (!texts.length || !texts[0].pageContent) {
      throw new Error('Document has no content to embed after splitting')
    }

    console.log(`📊 Created ${texts.length} text chunks`)

    // Add document metadata to each chunk
    const docsWithMetaData = texts.map((text, index) => ({
      ...text,
      metadata: {
        videoId,
        videoTitle,
        document_id: documentId,
        chunkIndex: index,
        totalChunks: texts.length,
        source: url
      }
    }))

    console.log('📮 Generating embeddings and storing in vector database...')
    await vectorStore.addDocuments(docsWithMetaData)

    console.log('✅ Document successfully processed and stored!')

    return {
      ok: true,
      message: `Successfully processed video "${videoTitle}" with ${texts.length} chunks`,
      chunksCreated: texts.length,
      videoTitle
    }

  } catch (error) {
    console.error('❌ storeDocument Error:', error.message)

    return {
      ok: false,
      error: error.message,
      suggestion: 'Try using a video with captions enabled (educational videos usually have them).'
    }
  }
}