name := """F36161DF3FCAB001C272FD10495FAFAF""".toLowerCase
organization := "org.mystic"

version := "0.1-SNAPSHOT"

scalaVersion := "2.12.2"

libraryDependencies += "com.typesafe.akka" %% "akka-actor" % "2.5.1"
libraryDependencies += "org.json4s" %% "json4s-native" % "3.5.2"

libraryDependencies += "com.typesafe.akka" %% "akka-testkit" % "2.5.1" % "test"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.3" % "test"
