// import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase'
// import { createSupabaseClient } from '../helpers/supabaseClient.js'
// import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
// import { YoutubeLoader } from '@langchain/community/document_loaders/web/youtube'
// import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'
// // import { v4 as uuidv4 } from 'uuid'

// export async function storeDocument(req) {
//   try {
//     if (!req?.body?.url) {
//       throw new Error('URL is required in the request body')
//     }

//     const { url, documentId } = req.body
//     const supabase = createSupabaseClient()

//     const embeddings = new GoogleGenerativeAIEmbeddings({
//       model: 'embedding-001' // ✅ Safe default
//     })

//     const vectorStore = new SupabaseVectorStore(embeddings, {
//       client: supabase,
//       tableName: 'embedded_documents',
//       queryName: 'match_documents'
//     })

//     // ✅ Await loader creation
//     const loader = await YoutubeLoader.createFromUrl(url, {
//       addVideoInfo: true
//     })

//     const docs = await loader.load()

//     if (docs[0]) {
//       docs[0].pageContent = `Video title: ${docs[0].metadata.title} | Video context: ${docs[0].pageContent}`
//     }

//     const textSplitter = new RecursiveCharacterTextSplitter({
//       chunkSize: 1000,
//       chunkOverlap: 200
//     })

//     const texts = await textSplitter.splitDocuments(docs)

//     if (!texts.length || !texts[0].pageContent) {
//       throw new Error('Document has no content to embed.')
//     }

//     const docsWithMetaData = texts.map((text) => ({
//       ...text,
//       metadata: {
//         ...(text.metadata || {}),
//         documentId
//       }
//     }))

//     await vectorStore.addDocuments(docsWithMetaData)
//   } catch (error) {
//     console.error('❌ storeDocument Error:', error.message)
//   }

//   return {
//     ok: true
//   }
// }
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase'
import { createSupabaseClient } from '../helpers/supabaseClient.js'
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai'
import { YoutubeLoader } from '@langchain/community/document_loaders/web/youtube'
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

export async function storeDocument(req) {
  try {
    if (!req?.body?.url) {
      throw new Error('URL is required in the request body')
    }

    const { url, documentId } = req.body
    console.log(`📥 Processing document: ${url}`)
    
    const supabase = createSupabaseClient()
    
    // ✅ Add API key to embeddings configuration
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: 'embedding-001',
      apiKey: process.env.GEMINI_API_KEY // ← This was missing!
    })

    // Verify API key is available
    if (!process.env.GEMINI_API_KEY) {
      throw new Error('GEMINI_API_KEY is not configured in environment variables')
    }

    const vectorStore = new SupabaseVectorStore(embeddings, {
      client: supabase,
      tableName: 'embedded_documents',
      queryName: 'match_documents'
    })

    console.log('🎥 Loading YouTube video...')
    const loader = await YoutubeLoader.createFromUrl(url, {
      addVideoInfo: true
    })
    
    const docs = await loader.load()
    
    if (!docs || docs.length === 0) {
      throw new Error('No content could be loaded from the YouTube video')
    }

    // Enhance the content with video metadata
    if (docs[0]) {
      docs[0].pageContent = `Video title: ${docs[0].metadata.title} | Video context: ${docs[0].pageContent}`
      console.log(`📝 Loaded video: "${docs[0].metadata.title}"`)
    }

    console.log('✂️ Splitting document into chunks...')
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200
    })
    
    const texts = await textSplitter.splitDocuments(docs)
    
    if (!texts.length || !texts[0].pageContent) {
      throw new Error('Document has no content to embed after splitting')
    }

    console.log(`📊 Created ${texts.length} text chunks`)

    // Add document metadata to each chunk
    const docsWithMetaData = texts.map((text, index) => ({
      ...text,
      metadata: {
        ...(text.metadata || {}),
        documentId,
        chunkIndex: index,
        totalChunks: texts.length
      }
    }))

    console.log('🔮 Generating embeddings and storing in vector database...')
    await vectorStore.addDocuments(docsWithMetaData)
    
    console.log('✅ Document successfully processed and stored!')
    
    return {
      ok: true,
      message: `Successfully processed video with ${texts.length} chunks`,
      chunksCreated: texts.length
    }

  } catch (error) {
    console.error('❌ storeDocument Error:', error.message)
    
    // Return error details for better debugging
    return {
      ok: false,
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    }
  }
}