package org.mystic

import akka.actor.{ActorSystem, Props}
import org.mystic.actors.{Client, Server, StorageActor}


/**
  * starting point
  */
object ActorApp extends App {

  val system = ActorSystem("f36")

  val server = system.actorOf(Props(classOf[Server], "localhost", 8080), "server")
  val client = system.actorOf(Props(classOf[Client], "localhost", 5555), "client")
  val storage = system.actorOf(Props(classOf[StorageActor]), "storage")

}
