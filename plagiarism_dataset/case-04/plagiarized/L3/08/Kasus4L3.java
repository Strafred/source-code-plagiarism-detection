public class Kasus4L3 {
    public static void main(String[] args) {
        int counter = 1;
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        while (counter <= 10) {
            System.out.println(counter + "\t\t" + counter * 1.609);
            counter++;
        }
    }
}
