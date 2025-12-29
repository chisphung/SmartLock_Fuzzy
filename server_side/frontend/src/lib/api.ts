import axios from 'axios';
import { CountResult } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export async function countPeopleFromFile(file: File, conf: number = 0.25): Promise<CountResult> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('conf', conf.toString());

  const response = await api.post<CountResult>('/api/v1/count/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

export async function countPeopleFromBase64(
  imageBase64: string,
  conf: number = 0.25
): Promise<CountResult> {
  const response = await api.post<CountResult>('/api/v1/count/base64', {
    image_base64: imageBase64,
    conf,
  });

  return response.data;
}

export async function countPeopleFromSource(
  source: string,
  conf: number = 0.25
): Promise<CountResult> {
  const response = await api.post<CountResult>('/api/v1/count', {
    source,
    conf,
  });

  return response.data;
}

export async function getResultImage(filename: string): Promise<string> {
  const response = await api.get(`/api/v1/result/${filename}`, {
    responseType: 'blob',
  });

  return URL.createObjectURL(response.data);
}

export async function healthCheck(): Promise<{ status: string; service: string }> {
  const response = await api.get('/health');
  return response.data;
}

export default api;
