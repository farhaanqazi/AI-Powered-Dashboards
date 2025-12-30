import axios from 'axios';

// Using relative paths for proxy in vite.config.js
const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('dataset', file);

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const loadExternalSource = async (source) => {
  const response = await api.post('/load_external', {
    external_source: source,
  });

  return response.data;
};

export const getDashboardData = async () => {
  const response = await api.get('/dashboard');
  return response.data;
};

export default api;