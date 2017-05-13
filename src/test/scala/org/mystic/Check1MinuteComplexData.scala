package org.mystic

import java.time.Instant
import java.time.temporal.ChronoUnit

import akka.actor.ActorSystem
import akka.pattern.ask
import akka.testkit.{DefaultTimeout, ImplicitSender, TestActorRef, TestKit}
import org.mystic.actors.StorageActor
import org.mystic.model._
import org.scalatest._

class Check1MinuteComplexData extends TestKit(ActorSystem("Check1MinData")) with DefaultTimeout with ImplicitSender
  with WordSpecLike with Matchers {

  val actorRef = TestActorRef(new StorageActor)

  val tradeTime = Instant.now().minus(1, ChronoUnit.MINUTES)
  val time = tradeTime.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  actorRef ? Deal(tradeTime.toEpochMilli, "A", 30.0d, 100)
  actorRef ? Deal(tradeTime.toEpochMilli, "B", 20.0d, 20)
  actorRef ? Deal(tradeTime.toEpochMilli, "C", 30.0d, 1)
  actorRef ? Deal(tradeTime.toEpochMilli, "A", 10.0d, 30)
  actorRef ? Deal(tradeTime.toEpochMilli, "B", 40.0d, 200)
  actorRef ? Deal(tradeTime.toEpochMilli, "C", 50.0d, 5)
  actorRef ? Deal(tradeTime.toEpochMilli, "A", 50.0d, 6)
  actorRef ? Deal(tradeTime.toEpochMilli, "B", 10.0d, 300)
  actorRef ? Deal(tradeTime.toEpochMilli, "C", 40.0d, 50)

  val future = actorRef ? AskFor1MData
  val res = future.value.get.get.asInstanceOf[Data1Minutes]
  res.data should have length 3

  res.data should contain(Some(CandleDeal("A", time, 30.0d, 50.0d, 10.0d, 50.0d, 136)))
  res.data should contain(Some(CandleDeal("B", time, 20.0d, 40.0d, 10.0d, 10.0d, 520)))
  res.data should contain(Some(CandleDeal("C", time, 30.0d, 50.0d, 30.0d, 40.0d, 56)))


}
