import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase'
import { createSupabaseClient } from '../helpers/supabaseClient.js'
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

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

// Helper function to get video title using oEmbed API
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

// Decode HTML entities
function decodeHtmlEntities(text) {
  if (!text) return ''
  return text
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&apos;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/\\n/g, ' ')
    .replace(/\n/g, ' ')
    .trim()
}

// Parse caption XML with multiple regex patterns
function parseCaptionXml(xml) {
  const texts = []
  
  // Pattern 1: <text start="..." dur="...">content</text>
  const pattern1 = /<text[^>]*>([^<]+)<\/text>/gi
  let match
  while ((match = pattern1.exec(xml)) !== null) {
    const decoded = decodeHtmlEntities(match[1])
    if (decoded) texts.push(decoded)
  }
  
  if (texts.length > 0) {
    console.log(`Pattern 1 matched: ${texts.length} segments`)
    return texts.join(' ')
  }
  
  // Pattern 2: Handle CDATA sections
  const pattern2 = /<text[^>]*><!\[CDATA\[(.*?)\]\]><\/text>/gi
  while ((match = pattern2.exec(xml)) !== null) {
    const decoded = decodeHtmlEntities(match[1])
    if (decoded) texts.push(decoded)
  }
  
  if (texts.length > 0) {
    console.log(`Pattern 2 (CDATA) matched: ${texts.length} segments`)
    return texts.join(' ')
  }
  
  // Pattern 3: JSON format (newer YouTube format)
  try {
    if (xml.includes('"events"')) {
      const jsonData = JSON.parse(xml)
      if (jsonData.events) {
        for (const event of jsonData.events) {
          if (event.segs) {
            for (const seg of event.segs) {
              if (seg.utf8) {
                const decoded = decodeHtmlEntities(seg.utf8)
                if (decoded && decoded !== '\n') texts.push(decoded)
              }
            }
          }
        }
      }
      if (texts.length > 0) {
        console.log(`Pattern 3 (JSON) matched: ${texts.length} segments`)
        return texts.join(' ')
      }
    }
  } catch (e) {
    // Not JSON format, continue
  }
  
  // Pattern 4: Try to extract any text between tags
  const pattern4 = />([^<]{2,})</g
  while ((match = pattern4.exec(xml)) !== null) {
    const decoded = decodeHtmlEntities(match[1])
    if (decoded && !decoded.startsWith('<?') && !decoded.includes('encoding')) {
      texts.push(decoded)
    }
  }
  
  if (texts.length > 0) {
    console.log(`Pattern 4 (generic) matched: ${texts.length} segments`)
    return texts.join(' ')
  }
  
  return null
}

// Fetch transcript from YouTube
async function fetchTranscript(videoId) {
  try {
    console.log('Fetching video page...')
    
    const watchUrl = `https://www.youtube.com/watch?v=${videoId}`
    const response = await fetch(watchUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
      }
    })
    
    if (!response.ok) {
      console.log(`Failed to fetch video page: ${response.status}`)
      return null
    }
    
    const html = await response.text()
    
    // Try to find captions in ytInitialPlayerResponse
    const playerResponseMatch = html.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\});/)
    
    if (playerResponseMatch) {
      try {
        const playerData = JSON.parse(playerResponseMatch[1])
        const captions = playerData?.captions?.playerCaptionsTracklistRenderer?.captionTracks
        
        if (captions && captions.length > 0) {
          console.log(`Found ${captions.length} caption track(s)`)
          
          // Prefer English, then any available
          let selectedTrack = captions.find(t => t.languageCode === 'en' && t.kind !== 'asr')
            || captions.find(t => t.languageCode === 'en')
            || captions.find(t => t.languageCode?.startsWith('en'))
            || captions[0]
          
          if (selectedTrack?.baseUrl) {
            const trackName = selectedTrack.name?.simpleText || selectedTrack.languageCode
            const isAutoGenerated = selectedTrack.kind === 'asr' ? ' (auto-generated)' : ''
            console.log(`Using caption track: ${trackName}${isAutoGenerated}`)
            
            // Fetch captions - try with fmt=json3 first (JSON format)
            let captionUrl = selectedTrack.baseUrl
            
            // Try JSON format first
            console.log('Trying JSON format (fmt=json3)...')
            const jsonUrl = captionUrl + '&fmt=json3'
            let captionResponse = await fetch(jsonUrl, {
              headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
              }
            })
            
            if (captionResponse.ok) {
              const captionData = await captionResponse.text()
              console.log(`Caption response length: ${captionData.length} chars`)
              
              // Try to parse as JSON
              try {
                const jsonCaptions = JSON.parse(captionData)
                if (jsonCaptions.events) {
                  const texts = []
                  for (const event of jsonCaptions.events) {
                    if (event.segs) {
                      for (const seg of event.segs) {
                        if (seg.utf8 && seg.utf8.trim() && seg.utf8 !== '\n') {
                          texts.push(seg.utf8.trim())
                        }
                      }
                    }
                  }
                  if (texts.length > 0) {
                    console.log(`✅ JSON format parsed: ${texts.length} segments`)
                    return texts.join(' ')
                  }
                }
              } catch (e) {
                console.log('JSON parse failed, trying XML format...')
              }
            }
            
            // Fallback to XML format
            console.log('Trying XML format...')
            captionResponse = await fetch(captionUrl, {
              headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
              }
            })
            
            if (captionResponse.ok) {
              const captionXml = await captionResponse.text()
              console.log(`Caption XML length: ${captionXml.length} chars`)
              console.log(`First 500 chars: ${captionXml.substring(0, 500)}`)
              
              const transcript = parseCaptionXml(captionXml)
              if (transcript) {
                return transcript
              }
            }
          }
        }
      } catch (e) {
        console.log('Failed to parse ytInitialPlayerResponse:', e.message)
      }
    }
    
    // Fallback: Try regex to find caption tracks
    const captionsRegex = /"captionTracks"\s*:\s*(\[[\s\S]*?\])\s*,\s*"/
    const captionMatch = html.match(captionsRegex)
    
    if (captionMatch) {
      try {
        const captionTracks = JSON.parse(captionMatch[1])
        if (captionTracks && captionTracks.length > 0) {
          console.log(`Found ${captionTracks.length} caption track(s) via regex`)
          
          const selectedTrack = captionTracks[0]
          if (selectedTrack?.baseUrl) {
            const captionResponse = await fetch(selectedTrack.baseUrl + '&fmt=json3')
            if (captionResponse.ok) {
              const captionData = await captionResponse.text()
              try {
                const jsonCaptions = JSON.parse(captionData)
                if (jsonCaptions.events) {
                  const texts = []
                  for (const event of jsonCaptions.events) {
                    if (event.segs) {
                      for (const seg of event.segs) {
                        if (seg.utf8 && seg.utf8.trim() && seg.utf8 !== '\n') {
                          texts.push(seg.utf8.trim())
                        }
                      }
                    }
                  }
                  if (texts.length > 0) {
                    console.log(`✅ Regex fallback parsed: ${texts.length} segments`)
                    return texts.join(' ')
                  }
                }
              } catch (e) {
                // Try XML parsing
                const transcript = parseCaptionXml(captionData)
                if (transcript) return transcript
              }
            }
          }
        }
      } catch (e) {
        console.log('Regex caption parse failed:', e.message)
      }
    }
    
    console.log('No captions could be extracted')
    return null
    
  } catch (error) {
    console.log('Transcript fetch error:', error.message)
    return null
  }
}

export async function storeDocument(req) {
  try {
    if (!req?.body?.url) {
      throw new Error('URL is required in the request body')
    }

    const { url, documentId } = req.body
    console.log(`📥 Processing document: ${url}`)

    const videoId = extractVideoId(url)
    if (!videoId) {
      throw new Error('Invalid YouTube URL. Could not extract video ID.')
    }

    console.log(`🎬 Video ID: ${videoId}`)

    const supabase = createSupabaseClient()

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

    const transcript = await fetchTranscript(videoId)

    if (!transcript || transcript.trim().length === 0) {
      throw new Error(
        'Could not fetch transcript for this video. ' +
        'The video may not have captions, or they may be disabled. ' +
        'Please try a different video with captions enabled.'
      )
    }

    console.log(`📄 Transcript length: ${transcript.length} characters`)

    const videoTitle = await getVideoTitle(videoId)
    console.log(`📝 Video Title: "${videoTitle}"`)

    const pageContent = `Video title: ${videoTitle} | Video context: ${transcript}`

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
      suggestion: 'Try using a video with captions enabled.'
    }
  }
}