FROM openjdk:13-jdk
RUN ["mkdir", "-p", "/static"]
ADD ./application/build/libs/*.jar app.jar
CMD ["java", "-jar", "app.jar"]