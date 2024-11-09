# Gunakan Node.js sebagai base image
FROM node:18.20.4

# Set working directory
WORKDIR /usr/src/app

# Salin file package.json dan package-lock.json
COPY package*.json ./

# Instal dependencies
RUN npm install

# Salin semua file proyek
COPY . .

# Expose port
EXPOSE 8080

# Jalankan aplikasi
CMD ["npm", "start"]
