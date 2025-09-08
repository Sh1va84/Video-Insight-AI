import { createClient } from '@supabase/supabase-js'

export const createSupabaseClient = () => {
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const  supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY


  if (!supabaseUrl || !supabaseAnonKey) {
    throw new Error('supabaseUrl and supabaseAnonKey are required.');
  }

  return createClient(supabaseUrl, supabaseAnonKey);
}
