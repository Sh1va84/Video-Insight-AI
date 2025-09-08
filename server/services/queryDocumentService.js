import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase'
import { createSupabaseClient } from '../helpers/supabaseClient.js'
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings
} from '@langchain/google-genai'
import {
  ChatPromptTemplate,
  MessagesPlaceholder
} from '@langchain/core/prompts'
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents'
import { createRetrievalChain } from 'langchain/chains/retrieval'
import { HumanMessage, AIMessage } from '@langchain/core/messages'
import { Readable } from 'stream'
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever'

export async function queryDocument (req) {
  try {
    const { conversationId, documentIds, query } = req.body
    const supabase = createSupabaseClient()

    // Store user query
    await supabase.from('conversation_messages').insert({
      conversation_id: conversationId,
      role: 'user',
      content: query
    })

    // Grab conversation history
    const { data: previousMessages } = await supabase
      .from('conversation_messages')
      .select('*')
      .eq('conversation_id', conversationId)
      .order('created_at', { ascending: false })
      .limit(14)

    // Initialise embedding models and LLM
    const embeddings = new GoogleGenerativeAIEmbeddings({
      model: 'embedding-001',
      apiKey: process.env.GEMINI_API_KEY
    })

    const llm = new ChatGoogleGenerativeAI({
      model: 'gemini-2.0-flash',
      apiKey: process.env.GEMINI_API_KEY,
      streamUsage: true
    })

    // Setup vector store configuration
    const vectorStoreConfig = {
      client: supabase,
      tableName: 'embedded_documents',
      queryName: 'match_documents'
    }

    // Only add filter if documentIds is provided and is not empty
    if (documentIds && Array.isArray(documentIds) && documentIds.length > 0) {
      vectorStoreConfig.filter = {
        document_id: documentIds
      }
    }
    
    // Initialise the vector store
    const vectorStore = new SupabaseVectorStore(embeddings, vectorStoreConfig)

    // Change the prompt based on query and documents
    const contextSystemPrompt =
      'Given a chat history and latest user question ' +
      'which might reference context in the chat history ' +
      'formulate a standalone question which can be understood ' +
      'without the chat history. DO NOT answer the question, ' +
      'just reformulate it if needed and otherwise return it as is.'

    // A set of instructions how to rewrite the question
    const prompt = ChatPromptTemplate.fromMessages([
      ['system', contextSystemPrompt],
      new MessagesPlaceholder('chat_history'),
      ['human', '{input}']
    ])

    // Retrieve the documents
    const retriever = vectorStore.asRetriever()
    const historyAwareRetriever = await createHistoryAwareRetriever({
      llm,
      retriever,
      rephrasePrompt: prompt
    })

    // Pass relevant documents to llm
    const systemPrompt =
      'You are an assistant for question answering tasks. ' +
      'Use the following pieces of retrieved context to answer ' +
      'the question. ' +
      '\n\n' +
      '{context}'

    const qaPrompt = ChatPromptTemplate.fromMessages([
      ['system', systemPrompt],
      new MessagesPlaceholder('chat_history'),
      ['human', '{input}']
    ])

    const qAChain = await createStuffDocumentsChain({
      llm,
      prompt: qaPrompt
    })

    const ragChain = await createRetrievalChain({
      retriever: historyAwareRetriever,
      combineDocsChain: qAChain
    })

    const history = (previousMessages || []).map(msg => {
      return msg.role === 'user'
        ? new HumanMessage(msg.content)
        : new AIMessage(msg.content)
    })

    // For streaming implementation
    if (req.headers['accept'] === 'text/event-stream') {
      const responseStream = new Readable({
        read () {}
      })

      let fullAnswer = ''

      // Start the streaming
      const stream = await ragChain.stream({
        input: query,
        chat_history: history
      })

      // Process each chunk
      ;(async () => {
        try {
          for await (const chunk of stream) {
            if (chunk.answer) {
              responseStream.push(
                `data: ${JSON.stringify({ content: chunk.answer })}\n\n`
              )
              fullAnswer += chunk.answer
            }
          }
          // End the stream
          responseStream.push(null)

          // Store AI response in database
          if (fullAnswer) {
            await supabase.from('conversation_messages').insert({
              conversation_id: conversationId,
              role: 'assistant',
              content: fullAnswer
            })
          }
        } catch (error) {
          console.error('Streaming error:', error)
          responseStream.push(
            `data: ${JSON.stringify({ error: 'Streaming error occurred' })}\n\n`
          )
          responseStream.push(null)
        }
      })()

      return responseStream
    } else {
      // For non-streaming response
      const response = await ragChain.invoke({
        input: query,
        chat_history: history
      })

      // Store AI response in database
      await supabase.from('conversation_messages').insert({
        conversation_id: conversationId,
        role: 'assistant',
        content: response.answer
      })

      return response.answer;
    }
  } catch (error) {
    console.error('❌ queryDocument Error:', error.message)
    throw error
  }
}